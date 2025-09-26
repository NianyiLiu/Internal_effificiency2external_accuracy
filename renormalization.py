import numpy as np
import scipy.linalg
import torch
import math
import os
import networkx as nx

from network_generation import write_weights
from data import Connectome
def laplacian_matrix(A):
    '''
    Information diffusion in complex networks rules as set out 
    by the Laplacian matrix L = D - A, where D is the degree
    matrix and A is the adjacency matrix of the undirected network.
    
    The Laplacian thus regulates the evolution of information of 
    a given initial specific state of the network s(0), which 
    will evolve with time as s(t) = exp(-t * L)s(0), where L is 
    the Laplacian matrix of the network.
    '''
    # Compute the degree matrix=
    D = torch.diag_embed(torch.sum(A, dim=-1))
    
    # Compute the Laplacian matrix
    L = D - A
    return L

def network_propagator(A, tau):
    '''
    The network propagator K represents the discrete counterpart 
    of the path-integral formulation of general diffusion processes, 
    and each matrix element substantially accounts for the sum of 
    diffusion trajectories along all possible paths connecting 
    nodes i and j up to a temporal scale tau.
    '''
    # Compute the Laplacian matrix
    L = laplacian_matrix(A)

    # Compute the propagator
    K = torch.matrix_exp(-tau * L)

    return K

# def operator(A, tau):
#     ''' In terms of the network propagator K now define the operator '''
#     K = network_propagator(A, tau)
#     rho = K / torch.trace(K)
#     return rho

# def operator_eigenvalues(A, tau):
#     '''
#     The generic eigenvalue in (0, 1] gives the relative weight 
#     of the corresponding Laplacian eigenvector in the 
#     eigenvector decomposition of the network state s(t)
#     '''
#     L = laplacian_matrix(A)
    
#     # Compute the eigenvalues of the Laplacian matrix
#     lambdas = torch.linalg.eigvalsh(L)
    
#     # eigenvalues of the Operator are simply related to 
#     # the eigenvalues of the Laplacian matrix
#     e = torch.exp(-tau * lambdas)
#     mu = e / torch.sum(e)
    
#     return mu

# def shannon_entropy(mu):
#     N = len(mu)
#     S = - 1 / math.log(N) * torch.sum(mu * torch.log(mu))
#     return S

# def entropy_derivative(A, tau):
#     ''' The derivative of the entropy concerning the (logarithm of the) diffusion time tau '''
#     tau = torch.tensor(tau, requires_grad=True)
    
#     # Compute the eigenvalues of the Operator
#     mu = operator_eigenvalues(A, tau)
    
#     # Compute the Shannon entropy
#     S = shannon_entropy(mu)
    
#     # Compute the derivative of the entropy concerning the logarithm of tau
#     derivative = torch.autograd.grad(S, tau)[0]
#     C = - tau * derivative
    
#     return C.item()

def operator_eigenvalues(A, tau):
    '''
    The generic eigenvalue in (0, 1] gives the relative weight 
    of the corresponding Laplacian eigenvector in the 
    eigenvector decomposition of the network state s(t)
    '''
    L = laplacian_matrix(A)
    
    # Compute the eigenvalues of the Laplacian matrix
    lambdas = torch.linalg.eigvalsh(L)
    
    # eigenvalues of the Operator are simply related to 
    # the eigenvalues of the Laplacian matrix
    # e = torch.exp(-tau * lambdas) # 1, 1
    e = torch.exp(-torch.einsum('i,j->ij', tau, lambdas))
    mu = e / torch.sum(e, -1, keepdim=True)
    return mu

def shannon_entropy(mu):
    N = mu.shape[-1]
    S = - 1 / math.log(N) * torch.sum(mu * torch.log(mu), -1)
    return S

def entropy_derivative(A, tau, derivative_only=True):
    ''' The derivative of the entropy concerning the (logarithm of the) diffusion time tau '''
    tau.requires_grad = True
    
    # Compute the eigenvalues of the Operator
    mu = operator_eigenvalues(A, tau)
    
    # Compute the Shannon entropy
    S = shannon_entropy(mu)
    
    # Compute the derivative of the entropy concerning the logarithm of tau
    derivative = torch.autograd.grad(S.sum(), tau)[0]
    C = - tau * derivative
    
    # return C.item()
    if derivative_only:
        return C
    return S, C

def entropy_derivative_tasks(start_n, stop_n, c_id,):
    float_type = torch.float64
    connectome = Connectome()
    disc = connectome.conn_name(c_id)
    print('Renormalization, derivative of the entropy of fc.')
    print(f'{disc}')
    for rand_id in range(start_n, stop_n):
        print(f'{rand_id} begin.', end=' ')
        output_dir = os.path.join('results', 'neurogym', disc, str(rand_id))
        fc_alphas = np.load(os.path.join(output_dir, f'fc_tasks_{rand_id}.npy'))
        
        s_alphas = []
        c_alphas = []
        for fc_tasks in fc_alphas:
            s_tasks = []
            c_tasks = []
            for fc in fc_tasks:
                m = torch.tensor(fc, dtype=float_type)
                tau_search = torch.pow(torch.tensor(10., dtype=float_type), torch.linspace(-4, 4, 401, dtype=float_type))
                s, c = entropy_derivative(m, tau_search, derivative_only=False)
                s_tasks.append(s.detach().numpy())
                c_tasks.append(c.detach().numpy())
            s_alphas.append(s_tasks)
            c_alphas.append(c_tasks)
            
        np.save(os.path.join(output_dir, 'entropy_derivative'), c_alphas)
        np.save(os.path.join(output_dir, 'entropy'), s_alphas)
        print(f'{rand_id} saved to {output_dir}.')
        
def entropy_derivative_networks(start_n, stop_n, c_id,):
    float_type = torch.float64
    alphas = (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
    connectome = Connectome()
    disc = connectome.conn_name(c_id)
    print('Renormalization, derivative of the entropy of sc.')
    print(f'{disc}')
    s_list = []
    c_list = []
    for rand_id in range(start_n, stop_n):
        print(f'{rand_id} begin.')
        w = connectome.conn(c_id, rand_id)
        if c_id == 2:
            raw_data = w
            n_nodes = len(raw_data)
            weights = raw_data[np.triu_indices(n_nodes, 1)][raw_data[np.triu_indices(n_nodes, 1)].nonzero()]
            for _ in range(10): np.random.shuffle(weights)
            w = raw_data > 0
            raw_graph = nx.from_numpy_array(w)
            write_weights(raw_graph, weights) # 把权重写到网络里
            w = nx.to_numpy_array(raw_graph)
        s_alphas = []
        c_alphas = []
        for a in alphas:
            mat = Connectome.set_spectrum_radius(w, a)
            m = torch.tensor(mat, dtype=float_type)
            tau_search = torch.pow(torch.tensor(10., dtype=float_type), torch.linspace(-4, 4, 401, dtype=float_type))
            s, c = entropy_derivative(m, tau_search, derivative_only=False)
            s_alphas.append(s.detach().numpy())
            c_alphas.append(c.detach().numpy())
        s_list.append(s_alphas)
        c_list.append(c_alphas)
    np.savez(os.path.join('results', 'neurogym', f'network_entropy_and_derivative_{disc}'), s=s_list, c=c_list)
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Information, RC model.')
    # parser.add_argument('-i', '--model_index', type=int, help='model index')
    parser.add_argument('--start_number', type=int, help='start number')
    parser.add_argument('--stop_number', type=int, help='stop number')
    
    models = [2, 32, 53, 51, 52]
    
    args = parser.parse_args()
    # i = args.model_index # 0到4
    start_number = args.start_number
    stop_number = args.stop_number
    
    # entropy_derivative_tasks(start_n=start_number, stop_n=stop_number, c_id=models[i])
    for c_id in models:
        entropy_derivative_tasks(start_n=start_number, stop_n=stop_number, c_id=c_id)
        entropy_derivative_networks(start_n=start_number, stop_n=stop_number, c_id=c_id)