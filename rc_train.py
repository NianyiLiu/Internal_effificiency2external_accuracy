import warnings
import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.base import is_classifier
import time
import datetime
import argparse
from conn2res.tasks import NeuroGymTask, ReservoirPyTask, Conn2ResTask
from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout
from conn2res import readout

from data_connectome import Connectome
from analysis_rc import res_task, NEUROGYM_TASKS, RESERVOIRPY_TASKS, CONN2RES_TASKS

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Run RC')
    parser.add_argument('--task_type', type=str, help='task type')
    parser.add_argument('--n_runs', type=int, help='number of runs for each task', default=1)
    parser.add_argument('--noise', type=float, help='noise sigma', default=0.)
    parser.add_argument('-c', '--connectome', type=int, help='connectome')
    parser.add_argument('--atlas', type=str, help='atlas', default='BNA')
    parser.add_argument('--exponential', action='store_true', help='use exponential alphas') # default False
    parser.add_argument('--save_w_in', action='store_true', help='save w_in only') # default False
    # parser.add_argument('--save_models', action='store_true', help='save w_in and w_out') # default False
    # parser.add_argument('--raw_mat', action='store_true', help='use raw matrix') # default False
    parser.add_argument('--start_number', type=int, help='start number')
    parser.add_argument('--stop_number', type=int, help='stop number')
    args = parser.parse_args()
    print('Connectome:', args.connectome)
    print('EXP:', args.exponential)
    
    noise_sigma = args.noise * np.sqrt(2)
    n_runs = args.n_runs
    # save_models = args.save_models
    # raw_mat = args.raw_mat
    exponential = args.exponential
    task_type = args.task_type
    
    if task_type == 'neurogym':
        tasks = NEUROGYM_TASKS
        metrics = ['balanced_accuracy_score', 'f1_score',]
    elif task_type == 'reservoirpy':
        tasks = RESERVOIRPY_TASKS
        metrics = ['mean_absolute_error', 'root_mean_squared_error', 'corrcoef']
    elif task_type == 'conn2res':
        tasks = CONN2RES_TASKS
        metrics = ['corrcoef', ]
    else:
        raise ValueError('task_type must be one of the following: neurogym, reservoirpy, conn2res')
    
    print('Using group matrix.')
    if exponential:
        alphas_exp = np.hstack([np.arange(-3, -1.0, 0.3), np.arange(-1.0, -0.5, 0.1), np.arange(-0.5, 0.5, 0.05), np.arange(0.5, 1.0, 0.1), np.linspace(1, 3, 21)]).round(2)
        alphas = np.exp(alphas_exp)
    else:
        alphas = tuple(i / 20 for i in range(15, 26))
    print('Using alphas', alphas)
    
    c = Connectome(connectivity=args.atlas)
    start_time = time.time()
    for subject_id in range(args.start_number, args.stop_number):
        w, disc = c.get_connectivity(args.connectome, subject_id)
        output_dir = os.path.join('results', task_type, disc, str(subject_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print('Subject id:', subject_id)
        print('Network discription:', disc)
    
        # scale
        w = (w - w.min()) / (w.max() - w.min())
        # normalize
        w = w / np.abs(scipy.linalg.eigh(w)[0]).max()
    
        input_labels, output_labels = c.get_labels()
        input_nodes, output_nodes = np.where(input_labels)[0], np.where(output_labels)[0]
        n_features = 41
        if os.path.exists(os.path.join(output_dir, 'w_in.npy')):
            w_in_all = np.load(os.path.join(output_dir, 'w_in.npy'))
        else:
            w_in_all = np.zeros((n_features, len(w)))
            w_in_all[:, input_nodes] = scipy.stats.ortho_group.rvs(max(n_features, len(input_nodes)))[0: n_features, 0: len(input_nodes)]
            if args.save_w_in:
                np.save(os.path.join(output_dir, 'w_in.npy'), w_in_all, )
        df_subj = []
        
        for task_name in tasks:
            print(f'TASK: {task_name}')
            task = res_task(task_name, task_type)
            activation = 'tanh'
            df_runs = []
            for run in range(n_runs):
                if isinstance(task, Conn2ResTask):
                    x, y = task.fetch_data(n_trials=4050, input_gain=0.0001)
                else:
                    x, y = task.fetch_data(n_trials=1000, input_gain=1)
                    if noise_sigma:
                        for xi in x:
                            xi += np.random.randn(*xi.shape) * noise_sigma
                x_train, x_test, y_train, y_test = readout.train_test_split(x, y)
                if task_type == 'neurogym' and task_name in (tasks[19], tasks[20]):
                    w_in = w_in_all[1:task.n_features+1]
                else:
                    w_in = w_in_all[:task.n_features]
                esn = EchoStateNetwork(w=w, activation_function=activation)
                readout_module = Readout(estimator=readout.select_model(y))
                df_alpha = []
                for i in range(len(alphas)):
                    alpha = alphas[i]
                    esn.w = alpha * w
                    rs_train = esn.simulate(
                        ext_input=x_train, w_in=w_in,
                        output_nodes=output_nodes
                    )
                    rs_test = esn.simulate(
                        ext_input=x_test, w_in=w_in,
                        output_nodes=output_nodes
                    )
                    df_res = readout_module.run_task(
                        X=(rs_train, rs_test), y=(y_train, y_test),
                        sample_weight='both', metric=metrics,
                        readout_modules=None, readout_nodes=None,
                    )
                    if exponential:
                        df_res['alpha'] = alpha
                        df_res['alpha exponent'] = alphas_exp[i]
                    else:
                        df_res['alpha'] = np.round(alpha, 3)
                    df_alpha.append(df_res)
                df_alpha = pd.concat(df_alpha, ignore_index=True)
                df_alpha['run'] = run
                df_alpha['activation'] = activation
                if 'module' in df_alpha.columns:
                    df_runs.append(
                        df_alpha[['module', 'n_nodes', 'activation', 'run', 'alpha'] + metrics]
                    )
                else:
                    df_runs.append(df_alpha[['activation', 'run', 'alpha'] + metrics])
            df_runs = pd.concat(df_runs, ignore_index=True)
            df_runs['task'] = task_name
            df_subj.append(df_runs)
        df_subj = pd.concat(df_subj, ignore_index=True)
        csv_name = f'alphas_performance.csv'
        df_subj.to_csv(
            os.path.join(output_dir, csv_name),
            index=False
        )
        print(f'subject{subject_id} done!')
        print(f'Elapsed time: {time.time() - start_time} ({datetime.datetime.today()})')