import argparse
import numpy as np
import scipy.stats
import pandas as pd
import bct
import seaborn as sns

import os

from data_connectome import Connectome
from performance_figure import save_fig

from conn2res import reservoir, connectivity
from conn2res.tasks import NeuroGymTask, ReservoirPyTask, Conn2ResTask

connectome = Connectome()
NEUROGYM_TASKS = (
    'AntiReach',
    'ContextDecisionMaking',
    'DelayComparison',
    'DelayMatchCategory',
    'DelayMatchSample',
    'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation',
    'DualDelayMatchSample',
    'GoNogo',
    'HierarchicalReasoning',
    'IntervalDiscrimination',
    'MotorTiming',
    'MultiSensoryIntegration',
    'OneTwoThreeGo',
    'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse',
    'PostDecisionWager',
    'ProbabilisticReasoning',
    'PulseDecisionMaking',
    'Reaching1D',
    'Reaching1DWithSelfDistraction',
    'ReadySetGo',
    'SingleContextDecisionMaking',
)
RESERVOIRPY_TASKS = (
    'henon_map',
    'lorenz',
    'multiscroll',
    'doublescroll',
    'rabinovich_fabrikant',
    'lorenz96',
    'rossler'
)
CONN2RES_TASKS = (
    'MemoryCapacity',
)
def res_task(task_name, task_type: str):
    if task_type == 'neurogym':
        return NeuroGymTask(name=task_name)
    elif task_type == 'reservoirpy':
        return ReservoirPyTask(name=task_name)
    elif task_type == 'conn2res':
        return Conn2ResTask(name=task_name)
    else:
        raise ValueError('task_type must be one of the following: neurogym, reservoirpy, conn2res')
    
# palette_index = np.array([1, 7, 3, 5, 9], dtype=int)
palette_index = np.array([1, 7, 9, 3, 5], dtype=int)
palette_light = [sns.color_palette('Paired')[i] for i in palette_index - 1]
palette_deep = [sns.color_palette('Paired')[i] for i in palette_index]
network_abbr = ['Vis', 'Sm', 'Da', 'Va', 'Lim', 'Fp', 'Dmn']
# from torch.nn.utils.rnn import pad_sequence

# #####################################################################
# First, let's initialize some constant variables
# #####################################################################
# project and figure directory
# PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PROJ_DIR = os.getcwd()
# OUTPUT_DIR = os.path.join(PROJ_DIR, 'examples', 'figs')
# if not os.path.isdir(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# name of the tasks to be performed
# TASKS = [
#     'PerceptualDecisionMaking',
#     'ContextDecisionMaking',
# ]
# https://neurogym.github.io/envs/index.html
NEUROGYM_TASKS = [
    'AntiReach', # 33 dimensions, index 0 = fixation
    # 'Bandit',  # *
    'ContextDecisionMaking', # 7 dimensions, index 0 fixation
    # 'DawTwoStep',  # *
    'DelayComparison', # 2 dimensions, index 0 fixation
    'DelayMatchCategory', # 3 dimensions, index 0 fixation
    'DelayMatchSample',
    'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation',
    # 'Detection',  # *
    'DualDelayMatchSample',
    # 'EconomicDecisionMaking',  # *
    'GoNogo',
    'HierarchicalReasoning',
    'IntervalDiscrimination',
    'MotorTiming', #
    'MultiSensoryIntegration',
    # 'Null',
    'OneTwoThreeGo',
    'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse',
    'PostDecisionWager',
    'ProbabilisticReasoning',
    'PulseDecisionMaking',
    'Reaching1D', # 32 dimensions, no fixation
    'Reaching1DWithSelfDistraction', # 32 dimensions, no fixation
    # 'ReachingDelayResponse',
    'ReadySetGo',
    'SingleContextDecisionMaking',
    # 'SpatialSuppressMotion',
    # 'ToneDetection'  # *
]
NEUROGYM_tasks_abbr = (
    'Anti Resp',
    'CDDM',
    'DC',
    'DMC',
    'DMS',
    'D1DDMS',
    'DPA',
    'DDMS',
    'Go Nogo',
    'HR', 
    'ID',
    'MT',
    'MSI',
    'OTT Go',
    'PDM',
    'PDMDR',
    'PDW',
    'PR',
    'PBDM',
    'R2S',
    'RSD',
    'RS Go',
    'SCDDM',
)
NEUROGYM_tasks_name = (
    'Anti Reach',
    'Context Decision Making',
    'Delay Comparison',
    'Delay Match Category',
    'Delay Match Sample',
    'Delay Match Sample Distractor 1D',
    'Delay Paired Association',
    'Dual Delay Match Sample',
    'Go Nogo',
    'Hierarchical Reasoning',
    'Interval Discrimination',
    'Motor Timing',
    'Multi Sensory Integration',
    'One Two Three Go',
    'Perceptual Decision Making',
    'Perceptual Decision Making Delay Response',
    'Post Decision Wager',
    'Probabilistic Reasoning',
    'Pulse Decision Making',
    'Reaching 1D',
    'Reaching 1D With Self Distraction',
    'Ready Set Go',
    'Single Context Decision Making',
)
NEUROGYM_tasks_discription = (
    'Anti-response task. During the fixation period, the agent fixates on a fixation point. During the following stimulus period, the agent is then shown a stimulus away from the fixation point. Finally, the agent needs to respond in the opposite direction of the stimulus during the decision period.',
    'Context-dependent decision-making task. The agent simultaneously receives stimulus inputs from two modalities ( for example, a colored random dot motion pattern with color and motion modalities). The agent needs to make a perceptual decision based on only one of the two modalities, while ignoring the other. The relevant modality is explicitly indicated by a rule signal.',
    'Delayed comparison. The agent needs to compare the magnitude of two stimuli are separated by a delay period. The agent reports its decision of the stronger stimulus during the decision period.',
    'Delayed match-to-category task. A sample stimulus is shown during the sample period. The stimulus is characterized by a one-dimensional variable, such as its orientation between 0 and 360 degree. This one-dimensional variable is separated into two categories (for example, 0-180 degree and 180-360 degree). After a delay period, a test stimulus is shown. The agent needs to determine whether the sample and the test stimuli belong to the same category, and report that decision during the decision period.',
    'Delayed match-to-sample task. A sample stimulus is shown during the sample period. The stimulus is characterized by a one-dimensional variable, such as its orientation between 0 and 360 degree. After a delay period, a test stimulus is shown. The agent needs to determine whether the sample and the test stimuli are equal, and report that decision during the decision period.',
    'Delayed match-to-sample with multiple, potentially repeating distractors. A sample stimulus is shown during the sample period. The stimulus is characterized by a one-dimensional variable, such as its orientation between 0 and 360 degree. After a delay period, the first test stimulus is shown. The agent needs to determine whether the sample and this test stimuli are equal. If so, it needs to produce the match response. If the first test is not equal to the sample stimulus, another delay period and then a second test stimulus follow, and so on.',
    'Delayed paired-association task. The agent is shown a pair of two stimuli separated by a delay period. For half of the stimuli-pairs shown, the agent should choose the Go response. The agent is rewarded if it chose the Go response correctly.',
    'Two-item Delay-match-to-sample. The trial starts with a fixation period. Then during the sample period, two sample stimuli are shown simultaneously. Followed by the first delay period, a cue is shown, indicating which sample stimulus will be tested. Then the first test stimulus is shown and the agent needs to report whether this test stimulus matches the cued sample stimulus. Then another delay and then test period follows, and the agent needs to report whether the other sample stimulus matches the second test stimulus.',
    'Go/No-go task. A stimulus is shown during the stimulus period. The stimulus period is followed by a delay period, and then a decision period. If the stimulus is a Go stimulus, then the subject should choose the action Go during the decision period, otherwise, the subject should remain fixation.',
    'Hierarchical reasoning of rules. On each trial, the subject receives two flashes separated by a delay period. The subject needs to judge whether the duration of this delay period is shorter than a threshold. Both flashes appear at the same location on each trial. For one trial type, the network should report its decision by going to the location of the flashes if the delay is shorter than the threshold. In another trial type, the network should go to the opposite direction of the flashes if the delay is short. The two types of trials are alternated across blocks, and the block transtion is unannouced.',
    'Comparing the time length of two stimuli. Two stimuli are shown sequentially, separated by a delay period. The duration of each stimulus is randomly sampled on each trial. The subject needs to judge which stimulus has a longer duration, and reports its decision during the decision period by choosing one of the two choice options.',
    'Agents have to produce different time intervals using different effectors (actions).',
    'Multi-sensory integration. Two stimuli are shown in two input modalities. Each stimulus points to one of the possible responses with a certain strength (coherence). The correct choice is the response with the highest summed strength from both stimuli. The agent is therefore encouraged to integrate information from both modalities equally.',
    'Agents reproduce time intervals based on two samples.',
    'Two-alternative forced choice task in which the subject has to integrate two stimuli to decide which one is higher on average. A noisy stimulus is shown during the stimulus period. The strength (coherence) of the stimulus is randomly sampled every trial. Because the stimulus is noisy, the agent is encouraged to integrate the stimulus over time.',
    'Perceptual decision-making with delayed responses. Agents have to integrate two stimuli and report which one is larger on average after a delay.',
    'Post-decision wagering task assessing confidence. The agent first performs a perceptual discrimination task (see for more details the PerceptualDecisionMaking task). On a random half of the trials, the agent is given the option to abort the sensory discrimination and to choose instead a sure-bet option that guarantees a small reward. Therefore, the agent is encouraged to choose the sure-bet option when it is uncertain about its perceptual decision.',
    'Probabilistic reasoning. The agent is shown a sequence of stimuli. Each stimulus is associated with a certain log-likelihood of the correct response being one choice versus the other. The final log-likelihood of the target response being, for example, option 1, is the sum of all log-likelihood associated with the presented stimuli. A delay period separates each stimulus, so the agent is encouraged to lean the log-likelihood association and integrate these values over time within a trial.',
    'Pulse-based decision making task. Discrete stimuli are presented briefly as pulses.',
    'Reaching to the stimulus. The agent is shown a stimulus during the fixation period. The stimulus encodes a one-dimensional variable such as a movement direction. At the end of the fixation period, the agent needs to respond by reaching towards the stimulus direction.',
    'Reaching with self distraction. In this task, the reaching state itself generates strong inputs that overshadows the actual target input. This task is inspired by behavior in electric fish where the electric sensing organ is distracted by discharges from its own electric organ for active sensing. Similar phenomena in bats.',
    'Agents have to measure and produce different time intervals. A stimulus is briefly shown during a ready period, then again during a set period. The ready and set periods are separated by a measure period, the duration of which is randomly sampled on each trial. The agent is required to produce a response after the set cue such that the interval between the response and the set cue is as close as possible to the duration of the measure period.',
    'Context-dependent decision-making task.The agent simultaneously receives stimulus inputs from two modalities ( for example, a colored random dot motion pattern with color and motion modalities). The agent needs to make a perceptual decision based on only one of the two modalities, while ignoring the other. The agent reports its decision during the decision period, with an optional delay period in between the stimulus period and the decision period. The relevant modality is not explicitly signaled.',
)
RESERVOIRPY_TASKS = [
    'henon_map',
    # 'logistic_map', # local variable 'sections' referenced before assignment
    'lorenz',
    # 'mackey_glass', # local variable 'sections' referenced before assignment
    'multiscroll',
    'doublescroll',
    'rabinovich_fabrikant',
    # 'narma', # local variable 'sections' referenced before assignment
    'lorenz96',
    'rossler'
]
CONN2RES_TASKS = [
    'MemoryCapacity'
]

# define metrics to evaluate readout's model performance
CLASS_METRICS = [
    # 'f1_score',
    # 'balanced_accuracy_score',
    'mean_absolute_error',
    # 'corrcoef',
]

# define alpha values to vary global reservoir dynamics
# ALPHAS = np.linspace(0, 2, 11)[1:]

# # select different activation functions to vary local dynamics
# ACT_FCNS = [
#     'tanh',
#     # 'sigmoid'
# ]

network_label_mapping = {
    'BNA_matrix_bfr': 'Empirical BNA ',
    'BNA_matrix_bfr_randmized': 'Randomly rewired BNA  ',
    'ER_BNA_bfr_density': 'ER network',
    'WS_BNA_bfr_density_sigma_shuffle': 'WS small-world',
    'BA_BNA_bfr_density': 'BA network',
}

def res_task(task_name, task_type: str):
    if task_type == 'neurogym':
        return NeuroGymTask(name=task_name)
    elif task_type == 'reservoirpy':
        return ReservoirPyTask(name=task_name)
    elif task_type == 'conn2res':
        return Conn2ResTask(name=task_name)
    else:
        raise ValueError('task_type must be one of the following: neurogym, reservoirpy, conn2res')

    
def dispach_data(data, types):
    unique_index = np.unique(types)
    data_list = []
    for i in unique_index:
        idx = np.where(types == i)[0]
        data_list.append([data[j] for j in idx])
    return data_list

def neurogym_task_input(n_features, n_trials, task_ids, task_type='neurogym', noise_sigma=0., pad_trials=False):
    # task_names = NEUROGYM_TASKS
    if noise_sigma:
        sigma = np.sqrt(2) * noise_sigma

    x_list = []
    # y_list = []
    task_idx = []
    n_features = 41
    if task_ids == None:
        task_ids = range(len(NEUROGYM_TASKS))
    elif isinstance(task_ids, int):
        task_ids = [task_ids]
    else:
        task_ids = list(task_ids)
    if isinstance(n_trials, int):
        n_trials = [n_trials for _ in range(len(task_ids))]
    for i, n in zip(task_ids, n_trials):
        task_name = NEUROGYM_TASKS[i]
        # task_name = NEUROGYM_TASKS[14]
        task = res_task(task_name, task_type)
        x, _ = task.fetch_data(n_trials=n, input_gain=1) # [batch size, time steps, features]
        if noise_sigma:
            for xi in x:
                xi += np.random.randn(*xi.shape) * sigma
        if pad_trials:
            max_len = max([x_instance.shape[0] for x_instance in x]) # x: (sequence length, input order)
        for j in range(len(x)):
            n_pad = n_features - x[j].shape[1]
            if pad_trials:
                seq_pad = max_len - x[j].shape[0]
            else:
                seq_pad = 0
            if i in (19, 20):
                x[j] = np.pad(x[j], ((0, seq_pad), (1, n_pad - 1)), 'constant', constant_values=0)
            else:
                x[j] = np.pad(x[j], ((0, seq_pad), (0, n_pad)), 'constant', constant_values=0)
        x_list.extend(x)
        task_idx.append(np.ones(len(x), dtype=int) * i)
        # task_samples.append(x[0].shape[1])
    task_idx = np.hstack(task_idx)
    # for x in x_list:
    #     print(x.shape)

    # for i in range(len(x_list)):
    #     n_pad = n_features - x_list[i].shape[1]
    #     if task_idx[i] in (19, 20):
    #         x_list[i] = np.pad(x_list[i], ((0, 0), (1, n_pad - 1)), 'constant', constant_values=0)
    #     else:
    #         x_list[i] = np.pad(x_list[i], ((0, 0), (0, n_pad)), 'constant', constant_values=0)
            
    # ext_input = utils.concat(x_list)
    # sections = utils.get_sections(x_list)
    return x_list, task_idx

def neurogym_input2hidden(x_list, task_idx, w, w_in, alpha, concat_method, output_nodes = np.arange(210)) -> list[list[np.ndarray]]:
    # tasks = [res_task(task_name, task_type) for task_name in task_names]

    conn = connectivity.Conn(w=w)
    conn.scale_and_normalize()
        
    esn = reservoir.EchoStateNetwork(w=conn.w, activation_function='tanh')
    esn.w = alpha * conn.w

    if concat_method == 'all':
        hidden_states_tasks = esn.simulate(
            ext_input=x_list, w_in=w_in,
            output_nodes=output_nodes
        )
        hidden_states_tasks = dispach_data(hidden_states_tasks, task_idx)
    elif concat_method == 'task':
        input_tasks = dispach_data(x_list, task_idx)
        hidden_states_tasks = []
        for x in input_tasks:
            hidden_states = esn.simulate(
                ext_input=x, w_in=w_in,
                output_nodes=output_nodes
            )
            hidden_states_tasks.append(hidden_states)
    elif concat_method == 'trial':
        raise NotImplementedError()
    else:
        raise ValueError()
    return hidden_states_tasks

def neurogym_hidden_states(w, alpha=1., concat_method='all', n_trials=10, task_ids=None, w_in=None):
    n_features = 41
    x_list, task_idx = neurogym_task_input(n_features=n_features, n_trials=n_trials, task_ids=task_ids)

    if w_in is None:
        input_nodes = np.arange(210, 246)
        w_in = np.zeros((n_features, 246))
        w_in[:, input_nodes] = scipy.stats.ortho_group.rvs(max(n_features, len(input_nodes)))[0: n_features, 0: len(input_nodes)]
    else:
        w_in = w_in

    hidden_states_tasks = neurogym_input2hidden(x_list, task_idx, w, w_in, alpha, concat_method)
    return hidden_states_tasks

def get_hidden_states_tasks(c, n, alphas, time_points=1500, pad_trials=False, concat_method='task', connectome = Connectome()) -> list[list[np.ndarray]]:
    
    task_mean_length = np.array([15.0, 19.004, 26.0, 27.0, 32.0, 53.0, 45.0, 40.0, 15.0, 37.81, 36.058, 58.824, 11.0, 58.262, 22.0, 42.166, 18.559, 34.481, 250.0, 10.0, 10.0, 43.925, 19.159])

    n_features = 41
    n_trials = (time_points / task_mean_length).round(0).astype(int)
    task_ids = None
    x_list, task_idx = neurogym_task_input(n_features=n_features, n_trials=n_trials, task_ids=task_ids, noise_sigma=0.01, pad_trials=pad_trials)
    w, disc = connectome.get_connectivity(c, n)
    w_in = np.load(os.path.join('results', 'neurogym', disc, str(n), 'w_in.npy'))

    h_list = []
    for a in alphas:
        hidden_states_tasks = neurogym_input2hidden(x_list=x_list, task_idx=task_idx, w=w, w_in=w_in, alpha=a, concat_method=concat_method, output_nodes=np.arange(246))
        if pad_trials:
            h = [np.stack(i, axis=1) for i in hidden_states_tasks]
        else:
            h = [np.vstack(i) for i in hidden_states_tasks]
        h_list.append(h)
    # output_path = os.path.join('results', 'neurogym', disc, str(n), 'hidden_state_tasks.npy')
    # np.save(output_path, h_list)
    return h_list

def simulated_data_frame(task_type, alpha = False, exmaple = True, model_name = 'BN_1065ind', start=0, stop=1065):
    ''' individual task performance '''
    sim_df = []
    # task_type = 'neurogym', 'conn2res', or 'reservoirpy'
    if exmaple:
        result_dir = 'examples'
        f_name_h = f'results_{task_type}' if not alpha else f'results_alphas_{task_type}'
    else:
        result_dir = os.path.join('results', task_type, model_name)
        f_name = 'performance.csv' if not alpha else 'alphas_performance.csv'
    for subject_id in range(start, stop):
        if exmaple:
            csv_name = os.path.join(result_dir, f'{f_name_h}_subject{subject_id}.csv')
        else:
            csv_name = os.path.join(result_dir, str(subject_id), f_name)
        df_tmp = pd.read_csv(csv_name)
        df_tmp['id'] = subject_id
        sim_df.append(df_tmp)
    # df_tmp['network'] = 'BNA_matrix_binary'
    sim_df = pd.concat(sim_df)
    # if isinstance(alpha, float) or isinstance(alpha, int):
    #     sim_df = sim_df[sim_df['alpha'] == alpha]
    return sim_df
def get_conn2res_task_metrics(task_type):
    if task_type == 'neurogym':
        metrics = 'balanced_accuracy_score'
    elif task_type == 'conn2res':
        metrics = 'corrcoef'
    elif task_type == 'reservoirpy':
        metrics = 'mean_absolute_error'
    return metrics
def simulated_mean_df(task_type, sim_df, labels=['id', 'task']):
    metrics = get_conn2res_task_metrics(task_type)
    tmp_df = sim_df.groupby(labels)[[metrics]].mean()
    return tmp_df
def simulated_data_array(task_type, sim_df):
    tmp_df = simulated_mean_df(task_type, sim_df)
    tmp_df = tmp_df.unstack()
    sim_data = tmp_df.to_numpy()
    # sim_task_name = tmp_df.columns.get_level_values(1).tolist()
    sim_task_name = tmp_df.columns.levels[1].tolist()
    return sim_data, sim_task_name
def simulated_alpha_rel(sim_df):
    return sim_df[(sim_df['run'] == 0) & (sim_df['task'] == sim_df['task'].values[0])][['alpha', 'id']]

def pad_tasks_sequence(hidden_states_tasks):
    hidden_states_tasks_padded = []
    for hidden_states in hidden_states_tasks:
        hidden_states_padded = pad_sequence(hidden_states)
        hidden_states_tasks_padded.append(hidden_states_padded)
    return hidden_states_tasks_padded

def pad_sequence(hidden_states):
    max_len = max([h.shape[0] for h in hidden_states])
    hidden_states_padded = []
    for h in hidden_states:
        n_pad = max_len - h.shape[0]
        h = np.pad(h, ((0, n_pad), (0, 0)), 'constant', constant_values=0)
        hidden_states_padded.append(h)
    return np.stack(hidden_states_padded, axis=1)

def readout_module(hidden_states_tasks_padded, w_out_tasks):
    y_pred_tasks = []
    for hidden_states, w_out in zip(hidden_states_tasks_padded, w_out_tasks):
        y_pred = np.einsum('ih,tjh->tijh', w_out, hidden_states)
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred_tasks.append(y_pred)
    return y_pred_tasks

def task_variance_modularity_sigma(c):
    from analysis_rc import small_world_sigma, task_variance
    connectome = Connectome()
    modularities = []
    sigmas = []
    for rand_id in range(50):
        w, disc = connectome.get_connectivity(c, random_id=rand_id)
        hidden_states_tasks = neurogym_hidden_states(w, concat_method='task')
        h_tv = [task_variance(np.stack(hidden_states, axis=1)) for hidden_states in hidden_states_tasks]
        h_tv = np.stack(h_tv, axis=1)
        
        mat = np.corrcoef(h_tv)
        np.fill_diagonal(mat, 0)
        # q = 90
        # percent = np.percentile(mat, q)
        # mat[mat < percent] = 0
        mat = bct.threshold_proportional(mat, 0.1)
        mat[mat.nonzero()] = 1
        modularities.append(bct.community_louvain(mat)[1])
        sigmas.append(small_world_sigma(mat, connected=False))
    print(disc)
    modularities = np.array(modularities)
    sigmas = np.array(sigmas)
    print(np.mean(modularities), np.mean(sigmas))
    np.save(os.path.join('results', f'{disc}_modularity.npy'), modularities)
    np.save(os.path.join('results', f'{disc}_sigma.npy'), sigmas)

def fc_tasks(start_n, stop_n, c_id):
    connectome = Connectome()
    alphas = np.array([0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0])
    disc = connectome.conn_name(c_id)
    print('FC for every task.')
    print(f'{disc}')
    for i in range(start_n, stop_n):
        print(f'{i} begin.', end=' ')
        h_lists = get_hidden_states_tasks(c_id, n=i, alphas=alphas, time_points=15000, connectome=connectome)
        fc_lists = [[np.corrcoef(h.T) for h in h_tasks] for h_tasks in h_lists]
        output_dir = os.path.join('results', 'neurogym', disc, str(i))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, f'fc_tasks_{i}'), fc_lists)
        print(f'{i} done.')

def fc(start_n, stop_n, c_id):
    connectome = Connectome()
    alphas = np.array([0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0])
    disc = connectome.conn_name(c_id)
    print('FC of all tasks in one sequence.')
    print(f'{disc}')
    for i in range(start_n, stop_n):
        print(f'{i} begin.', end=' ')
        h_lists = get_hidden_states_tasks(c_id, n=i, alphas=alphas, time_points=15000, connectome=connectome, concat_method='all')
        fc_lists = [np.corrcoef(np.vstack(h_l).T) for h_l in h_lists]
        output_dir = os.path.join('results', 'neurogym', disc, str(i))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, f'fc_{i}'), fc_lists)
        print(f'{i} done.')
        
def task_variance_graph(start_n, stop_n, c_id,):
    from analysis_rc import task_variance, small_world_sigma
    connectome = Connectome()
    alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
    disc = connectome.conn_name(c_id)
    print('Graph attributes of task variance.')
    print(f'{disc}')
    for rand_id in range(start_n, stop_n):
        print(f'{rand_id} begin.', end=' ')
        output_dir = os.path.join('results', 'neurogym', disc, str(rand_id))
        h_tv_alphas = np.load(os.path.join(output_dir, 'task_variance.npy'))
        
        mat_alphas = [np.corrcoef(t) for t in h_tv_alphas]
        for mat in mat_alphas:
            np.fill_diagonal(mat, 0)
            bct.threshold_proportional(mat, 0.1, copy=False)
            mat[mat.nonzero()] = 1
            
        communities = [bct.community_louvain(mat) for mat in mat_alphas]
        tmp_n_communities = [np.max(i[0]) for i in communities]
        tmp_modularities = [i[1] for i in communities]
        tmp_sigmas = [small_world_sigma(mat, connected=False)[0] for mat in mat_alphas]
        df_tmp = pd.DataFrame({
            'Modularity': tmp_modularities, 
            'Number of communities':tmp_n_communities,
            'Small-worldness': tmp_sigmas,
            'network': disc,
            'Spectrum radius': alphas,
        })

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_tmp.to_csv(os.path.join(output_dir, 'task_variance_graph_attr.csv'))
        print(f'{rand_id} saved to {output_dir}.')

def task_variance_only(start_n, stop_n, c_id):
    from analysis_rc import task_variance
    connectome = Connectome()
    alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
    disc = connectome.conn_name(c_id)
    print('Task variance only.')
    print(f'{disc}')
    for rand_id in range(start_n, stop_n):
        print(f'{rand_id} begin.', end=' ')
        h_lists = get_hidden_states_tasks(c_id, rand_id, alphas, pad_trials=True,connectome=connectome)
        h_tv_lists = [[task_variance(h) for h in hidden_states_list] for hidden_states_list in h_lists]
        h_tv_alphas = [np.stack(t, axis=1) for t in h_tv_lists]
    
        output_dir = os.path.join('results', 'neurogym', disc, str(rand_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, 'task_variance'), h_tv_alphas)
        print(f'{rand_id} saved to {output_dir}.')
        
def fc_tasks_graph(start_n, stop_n, c_id,):
    from analysis_rc import small_world_sigma
    connectome = Connectome()
    alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
    disc = connectome.conn_name(c_id)
    print('Graph attributes of fc, single task.')
    print(f'{disc}')
    for rand_id in range(start_n, stop_n):
        print(f'{rand_id} begin.', end=' ')
        output_dir = os.path.join('results', 'neurogym', disc, str(rand_id))
        fc_alphas = np.load(os.path.join(output_dir, f'fc_tasks_{rand_id}.npy'))
        
        for fc_tasks in fc_alphas:
            for mat in fc_tasks:
                np.fill_diagonal(mat, 0)
                bct.threshold_proportional(mat, 0.1, copy=False)
                mat[mat.nonzero()] = 1
        
        df = []
        for a, fc_tasks in zip(alphas, fc_alphas):
            communities = [bct.community_louvain(mat) for mat in fc_tasks]
            tmp_n_communities = [np.max(i[0]) for i in communities]
            tmp_modularities = [i[1] for i in communities]
            tmp_sigmas = [small_world_sigma(mat, connected=False)[0] for mat in fc_tasks]
            df_tmp = pd.DataFrame({
                'Modularity': tmp_modularities, 
                'Number of communities':tmp_n_communities,
                'Small-worldness': tmp_sigmas,
                'Network type': disc,
                'Task': NEUROGYM_tasks_abbr,
                'Spectrum radius': a,
            })
            df.append(df_tmp)
        df_tmp = pd.concat(df)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_tmp.to_csv(os.path.join(output_dir, 'fc_tasks_graph_attr.csv'))
        print(f'{rand_id} saved to {output_dir}.')
        
def fc_graph(start_n, stop_n, c_id,):
    from analysis_rc import small_world_sigma
    connectome = Connectome()
    alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
    disc = connectome.conn_name(c_id)
    print('Graph attributes of fc, all tasks.')
    print(f'{disc}')
    for rand_id in range(start_n, stop_n):
        print(f'{rand_id} begin.', end=' ')
        output_dir = os.path.join('results', 'neurogym', disc, str(rand_id))
        fc_alphas = np.load(os.path.join(output_dir, f'fc_{rand_id}.npy'))
        
        for mat in fc_alphas:
            np.fill_diagonal(mat, 0)
            bct.threshold_proportional(mat, 0.1, copy=False)
            mat[mat.nonzero()] = 1
        
        communities = [bct.community_louvain(mat) for mat in fc_alphas]
        tmp_n_communities = [np.max(i[0]) for i in communities]
        tmp_modularities = [i[1] for i in communities]
        tmp_sigmas = [small_world_sigma(mat, connected=False)[0] for mat in fc_alphas]
        df_tmp = pd.DataFrame({
            'Modularity': tmp_modularities, 
            'Number of communities':tmp_n_communities,
            'Small-worldness': tmp_sigmas,
            'Network type': disc,
            'Spectrum radius': alphas,
        })

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_tmp.to_csv(os.path.join(output_dir, 'fc_graph_attr.csv'))
        print(f'{rand_id} saved to {output_dir}.')

def gradients_tasks(start_n, stop_n, c_id):
    from brainspace.gradient import GradientMaps
    from scipy.optimize import linear_sum_assignment
    def corr2(a, b):
        return np.corrcoef(np.concatenate((a, b), axis=0))[:len(a), len(a):len(a)+len(b)]
    def maximum_mean_corr(grad1, grad2):
        corr_matrix = abs(corr2(grad1, grad2))
        # Use the linear_sum_assignment function to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(corr_matrix, maximize=True)
        # Calculate the maximum sum by summing the assigned elements
        return corr_matrix[row_ind, col_ind].mean()
    connectome = Connectome()
    # alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
    disc = connectome.conn_name(c_id)
    print('Gradients of acitivation correlation matrix.')
    for rand_id in range(start_n, stop_n):
        print(f'{rand_id} begin.', end=' ')
        fc_path = os.path.join('results', 'neurogym', disc, str(rand_id), f'fc_tasks_{rand_id}.npy')
        fc = np.load(fc_path, allow_pickle=True)
        grads = [] # 9, 23, 6, 246
        for a_i in range(fc.shape[0]):
            tasks_grad = []
            for t_i in range(fc.shape[1]):
                gm = GradientMaps(n_components=6, approach='dm', kernel='normalized_angle')
                gm.fit(fc[a_i][t_i])
                tasks_grad.append(gm.gradients_.T)
            grads.append(tasks_grad)
        # grads = np.asarray(grads)
        # grads.resize((fc.shape[0], fc.shape[1], 6, 246))
        
        mean_corr = []
        for tasks_grad in grads:
            mean_task_corr = np.zeros((len(tasks_grad), len(tasks_grad))) # 23, 23
            for i in range(len(tasks_grad)):
                for j in range(i+1, len(tasks_grad)):
                    mean_task_corr[i, j] = maximum_mean_corr(tasks_grad[i], tasks_grad[j])
            mean_corr.append(mean_task_corr)
        mean_corr = np.asarray(mean_corr) # 9, 23, 23
        
        output_dir = os.path.join('results', 'neurogym', disc, str(rand_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, 'gradients_tasks'), grads)
        np.save(os.path.join(output_dir, 'gradients_tasks_correlation'), mean_corr)
        print(f'{rand_id} saved to {output_dir}.')
        
def gradients_tasks_correlation(start_n, stop_n, c_id, n_comp=4):
    from scipy.optimize import linear_sum_assignment
    def corr2(a, b):
        return np.corrcoef(np.concatenate((a, b), axis=0))[:len(a), len(a):len(a)+len(b)]
    def maximum_mean_corr(grad1, grad2):
        corr_matrix = abs(corr2(grad1, grad2))
        # Use the linear_sum_assignment function to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(corr_matrix, maximize=True)
        # Calculate the maximum sum by summing the assigned elements
        return corr_matrix[row_ind, col_ind].mean()
    connectome = Connectome()
    # alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
    disc = connectome.conn_name(c_id)
    print('Gradients of acitivation correlation matrix.')
    for rand_id in range(start_n, stop_n):
        print(f'{rand_id} begin.', end=' ')
        output_dir = os.path.join('results', 'neurogym', disc, str(rand_id))
        grad_path = os.path.join(output_dir, 'gradients_tasks.npy')
        grads = np.load(grad_path, allow_pickle=True)[:, :, :n_comp] # (9, 23, 6, 246)
        
        mean_corr = []
        for tasks_grad in grads:
            mean_task_corr = np.zeros((len(tasks_grad), len(tasks_grad))) # 23, 23
            for i in range(len(tasks_grad)):
                for j in range(i+1, len(tasks_grad)):
                    mean_task_corr[i, j] = maximum_mean_corr(tasks_grad[i], tasks_grad[j])
            mean_corr.append(mean_task_corr)
        mean_corr = np.asarray(mean_corr) # 9, 23, 23
        
        np.save(os.path.join(output_dir, f'gradients_{n_comp}_tasks_correlation'), mean_corr)
        print(f'{rand_id} saved to {output_dir}.')
        
def gradients(start_n, stop_n, c_id):
    from brainspace.gradient import GradientMaps
    connectome = Connectome()
    # alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
    disc = connectome.conn_name(c_id)
    print('Gradients of acitivation correlation matrix.')
    for rand_id in range(start_n, stop_n):
        print(f'{rand_id} begin.', end=' ')
        fc_path = os.path.join('results', 'neurogym', disc, str(rand_id), f'fc_{rand_id}.npy')
        fc = np.load(fc_path, allow_pickle=True)
        grads = []
        for a_i in range(fc.shape[0]):
            gm = GradientMaps(n_components=6, approach='dm', kernel='normalized_angle')
            gm.fit(fc[a_i])
            grads.append(gm.gradients_.T)
        # grads = np.asarray(grads)
        # grads.resize((fc.shape[0], 6, 246))
        
        output_dir = os.path.join('results', 'neurogym', disc, str(rand_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, 'gradients'), grads)
        print(f'{rand_id} saved to {output_dir}.')
    
def pvalue_label(p):
    if p <= 0.0001:
    #     return '***'
        return '**'
    # elif p <= 0.001:
    #     return '**'
    # elif p <= 0.01:
    #     return '**'
    elif p <= 0.05:
        return '*'
    else:
        return ''
            
if __name__ == '__main__':
    models = [2, 32, 53, 51, 52]
    parser = argparse.ArgumentParser(description='Analysis RC')
    parser.add_argument('--func', type=str, default='', help='function to run')
    parser.add_argument('--start', type=int, default=0, help='start number')
    parser.add_argument('--stop', type=int, default=1000, help='stop number')
    parser.add_argument('-c', type=int, default=2, help='connectome id')
    args = parser.parse_args()
    func = args.func

    start_number = args.start
    stop_number = args.stop
    c = args.c
    assert c in models, f'c must be one of {models}'

    if func == 'fc_tasks':
        fc_tasks(start_n=start_number, stop_n=stop_number, c_id=c)
    elif func == 'fc':
        fc(start_n=start_number, stop_n=stop_number, c_id=c)
    elif func == 'task_variance_only':
        task_variance_only(start_n=start_number, stop_n=stop_number, c_id=c)
    elif func == 'task_variance_graph':
        task_variance_graph(start_n=start_number, stop_n=stop_number, c_id=c)
    elif func == 'gradients_tasks':
        gradients_tasks(start_n=start_number, stop_n=stop_number, c_id=c)
    elif func == 'gradients':
        gradients(start_n=start_number, stop_n=stop_number, c_id=c)
    elif func == 'fc_tasks_graph':
        fc_tasks_graph(start_n=start_number, stop_n=stop_number, c_id=c)
    elif func == 'fc_graph':
        fc_graph(start_n=start_number, stop_n=stop_number, c_id=c)
    elif func == 'gradients_tasks_correlation':
        gradients_tasks_correlation(start_n=start_number, stop_n=stop_number, c_id=c)
    elif not func:
        fc_tasks(start_n=start_number, stop_n=stop_number, c_id=c)
        fc(start_n=start_number, stop_n=stop_number, c_id=c)
        task_variance_only(start_n=start_number, stop_n=stop_number, c_id=c)
        task_variance_graph(start_n=start_number, stop_n=stop_number, c_id=c)
        gradients_tasks(start_n=start_number, stop_n=stop_number, c_id=c)
        gradients(start_n=start_number, stop_n=stop_number, c_id=c)
        fc_tasks_graph(start_n=start_number, stop_n=stop_number, c_id=c)
        fc_graph(start_n=start_number, stop_n=stop_number, c_id=c)
        gradients_tasks_correlation(start_n=start_number, stop_n=stop_number, c_id=c)
    else:
        raise ValueError(f"Unknown function: {func}")

    # for i in range(len(models)):
    #     fc_tasks(start_n=start_number, stop_n=stop_number, c_id=models[i])
    #     fc(start_n=start_number, stop_n=stop_number, c_id=models[i])
    #     task_variance_only(start_n=start_number, stop_n=stop_number, c_id=models[i])
    #     task_variance_graph(start_n=start_number, stop_n=stop_number, c_id=models[i])
    #     gradients_tasks(start_n=start_number, stop_n=stop_number, c_id=models[i])
    #     gradients(start_n=start_number, stop_n=stop_number, c_id=models[i])
    #     fc_tasks_graph(start_n=start_number, stop_n=stop_number, c_id=models[i])
    #     fc_graph(start_n=start_number, stop_n=stop_number, c_id=models[i])
    #     gradients_tasks_correlation(start_n=start_number, stop_n=stop_number, c_id=models[i])