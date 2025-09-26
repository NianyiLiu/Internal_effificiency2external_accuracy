import numpy as np
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx
import bct
import os
import sys

from data_connectome import Connectome

def write_weights(graph, weights):
    for _ in range(10): np.random.shuffle(weights)
    for i, e in enumerate(graph.edges):
        graph.edges[e]['weight'] = weights[i]
    return graph
def align_edge_number(graph, n_edges, check_connected=True):
    while graph.number_of_edges() < n_edges:
        graph.add_edge(*np.random.default_rng().choice(graph.nodes(), 2, replace=False))
    while graph.number_of_edges() > n_edges:
        graph.remove_edge(*np.random.default_rng().choice(graph.edges(), replace=False))
    if check_connected:
        assert nx.is_connected(graph), 'The network is not connected.'
    return graph

def generate_er_networks_bna_bfr(n = 1000, ):
    connectome = Connectome()
    raw_data = connectome.conn(2)
    weights = raw_data[np.triu_indices(246, 1)][raw_data[np.triu_indices(246, 1)].nonzero()]
    for _ in range(10): np.random.shuffle(weights)
    w = raw_data > 0
    raw_graph = nx.from_numpy_array(w)
    n_edges = raw_graph.number_of_edges()

    net_discription = 'ER_BNA_bfr_density'
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    os.makedirs(rand_net_dir, exist_ok=True)
    for net_id in range(n):
        # Generate a random network with the same average density as the original network
        graph = nx.fast_gnp_random_graph(246, w.sum()/(246*245))
        align_edge_number(graph, n_edges)
        write_weights(graph, weights) # Write weights to the network
        w_r = nx.to_numpy_array(graph)
        matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
        np.savetxt(matrix_file, w_r, delimiter=',')

def generate_ba_networks_bna(n = 1000):
    connectome = Connectome()
    raw_data = connectome.conn(1)
    raw_graph = nx.from_numpy_array(raw_data)
    n_edges = raw_graph.number_of_edges()

    net_discription = 'BA_BNA_binary_density'
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    os.makedirs(rand_net_dir, exist_ok=True)
    m = max([i for i in range(246 // 2) if n_edges - (246 - i) * i > 0]) - 1
    for net_id in range(n):
        print(net_id)
        graph = nx.extended_barabasi_albert_graph(246, m, 1 - (246-m)*m / n_edges, 0)
        w_r = nx.to_numpy_array(graph)
        node_index = np.arange(len(w_r))
        for _ in range(10): np.random.shuffle(node_index)
        w_r = w_r[node_index][:, node_index]
        matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
        np.savetxt(matrix_file, w_r, delimiter=',', fmt='%d')

def generate_ba_networks_bna_bfr(n = 1000):
    connectome = Connectome()
    raw_data = connectome.conn(2)
    weights = raw_data[np.triu_indices(246, 1)][raw_data[np.triu_indices(246, 1)].nonzero()]
    for _ in range(10): np.random.shuffle(weights)
    w = raw_data > 0
    raw_graph = nx.from_numpy_array(w)
    n_edges = raw_graph.number_of_edges()

    net_discription = 'BA_BNA_bfr_density'
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    os.makedirs(rand_net_dir, exist_ok=True)
    m = max([i for i in range(246 // 2) if n_edges - (246 - i) * i > 0]) - 1
    for net_id in range(n):
        print(net_id)
        graph = nx.extended_barabasi_albert_graph(246, m, 1 - (246-m)*m / n_edges, 0)
        align_edge_number(graph, n_edges)
        write_weights(graph, weights) # Write weights to the network
        w_r = nx.to_numpy_array(graph)
        node_index = np.arange(len(w_r))
        for _ in range(10): np.random.shuffle(node_index)
        w_r = w_r[node_index][:, node_index]
        matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
        np.savetxt(matrix_file, w_r, delimiter=',')

def generate_ws_networks_bna_sigma(n = 1000, shuffle = False, start_n = 0):
    ''' 
        Adjust parameters so that the small-world property of the network is similar to the sigma of the BNA binary matrix.
        Also shuffle the node indices.
    '''
    net_discription = 'WS_BNA_binary_density_sigma'
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    if shuffle: rand_net_dir += '_shuffle'
    os.makedirs(rand_net_dir, exist_ok=True)
    for net_id in range(start_n, start_n + n):
        p = np.random.rand() * 0.02 + 0.25 # This generation method is almost the same as the previous shuffled result. After many attempts, this method was chosen.
        # p = 0.26  # Similar to the above result
        # p = np.random.rand() * 0.11 + 0.21
        # p = 0.4  # The sigma of such a network is less than 0.1, but the performance is similar to BNA
        print(net_id, p)
        while True:
            try:
                graph = nx.connected_watts_strogatz_graph(246, 90, p)
                graph.remove_edges_from(np.random.default_rng().choice(graph.edges(), 10, False, axis=0)) # The network may become disconnected after removing edges
                assert nx.is_connected(graph), 'The WS network is not connected.'
            except AssertionError:
                print('The WS network is not connected.')
                print('Try again.')
            else:
                break
        w_r = nx.to_numpy_array(graph)
        if shuffle:
            node_index = np.arange(len(w_r))
            for _ in range(10): np.random.shuffle(node_index)
            w_r = w_r[node_index][:, node_index]
            matrix_file = os.path.join(rand_net_dir, f'{net_discription}_shuffle_{net_id}.csv')
        else:
            matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
        np.savetxt(matrix_file, w_r, delimiter=',', fmt='%d')

def generate_ws_networks_bna_bfr_sigma(n = 1000, shuffle = False, start_n = 0):
    ''' 
        Adjust parameters so that the small-world property of the network is similar to the sigma of the BNA bfr matrix.
        Also shuffle the node indices.
    '''
    connectome = Connectome()
    raw_data = connectome.conn(2)
    weights = raw_data[np.triu_indices(246, 1)][raw_data[np.triu_indices(246, 1)].nonzero()]
    for _ in range(10): np.random.shuffle(weights)

    net_discription = 'WS_BNA_bfr_density_sigma'
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    if shuffle: rand_net_dir += '_shuffle'
    os.makedirs(rand_net_dir, exist_ok=True)
    for net_id in range(start_n, start_n + n):
        p = np.random.rand() * 0.02 + 0.35 # This generation method is almost the same as the previous shuffled result. After many attempts, this method was chosen.
        # p = 0.26  # Similar to the above result
        # p = np.random.rand() * 0.11 + 0.21
        # p = 0.4  # The sigma of such a network is less than 0.1, but the performance is similar to BNA
        print(net_id, p)
        while True:
            try:
                graph = nx.connected_watts_strogatz_graph(246, 52, p)
                graph.remove_edges_from(np.random.default_rng().choice(graph.edges(), 174, False, axis=0)) # The network may become disconnected after removing edges
                assert nx.is_connected(graph), 'The WS network is not connected.'
            except AssertionError:
                print('The WS network is not connected.')
                print('Try again.')
            else:
                break
        graph = write_weights(graph, weights) # Write weights to the network
        w_r = nx.to_numpy_array(graph)
        if shuffle:
            node_index = np.arange(len(w_r))
            for _ in range(10): np.random.shuffle(node_index)
            w_r = w_r[node_index][:, node_index]
            matrix_file = os.path.join(rand_net_dir, f'{net_discription}_shuffle_{net_id}.csv')
        else:
            matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
        np.savetxt(matrix_file, w_r, delimiter=',')

def check_connected(net_dir, net_discription, n = 1000):
    for net_id in range(n):
        matrix_file = os.path.join(net_dir, f'{net_discription}_{net_id}.csv')
        w = np.loadtxt(matrix_file, delimiter=',')
        graph = nx.from_numpy_array(w)
        if not nx.is_connected(graph):
            print(f"{n} is not connected")
    print('Check over.')
    
# The following part of the code is the final version, suitable for generating randomized versions of any existing network.
def generate_randomize_networks(c, n = 100, start_n = 0, connectivity='human'):
    connectome = Connectome(connectivity)
    raw_data, disc = connectome.get_connectivity(c)

    net_discription = 'Randomize_' + disc
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    os.makedirs(rand_net_dir, exist_ok=True)
    itr = 10
    for net_id in range(start_n, start_n + n):
        print(net_id)
        w_r, eff = bct.reference.randmio_und_connected(raw_data, itr)
        print(f'Randomizing network with {eff} number of actual rewirings carried out.')
        matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
        np.savetxt(matrix_file, w_r, delimiter=',')

def generate_er_networks(c, n = 100, start_n = 0, connectivity='human'):
    connectome = Connectome(connectivity)
    raw_data, disc = connectome.get_connectivity(c)
    n_nodes = len(raw_data)
    weights = raw_data[np.triu_indices(n_nodes, 1)][raw_data[np.triu_indices(n_nodes, 1)].nonzero()]
    for _ in range(10): np.random.shuffle(weights)
    w = raw_data > 0
    raw_graph = nx.from_numpy_array(w)
    n_edges = raw_graph.number_of_edges()

    net_discription = 'ER_' + disc
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    os.makedirs(rand_net_dir, exist_ok=True)
    for net_id in range(start_n, start_n + n):
        print(net_id)
        graph = nx.fast_gnp_random_graph(n_nodes, w.sum()/(n_nodes*(n_nodes-1))) # Generate random network with same average density as the original network
        align_edge_number(graph, n_edges)
        write_weights(graph, weights) # Write weights to the network
        w_r = nx.to_numpy_array(graph)
        matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
        np.savetxt(matrix_file, w_r, delimiter=',')

def generate_ba_networks(c, n = 100, start_n = 0, connectivity='human'):
    connectome = Connectome(connectivity)
    raw_data, disc = connectome.get_connectivity(c)
    n_nodes = len(raw_data)
    weights = raw_data[np.triu_indices(n_nodes, 1)][raw_data[np.triu_indices(n_nodes, 1)].nonzero()]
    for _ in range(10): np.random.shuffle(weights)
    w = raw_data > 0
    raw_graph = nx.from_numpy_array(w)
    n_edges = raw_graph.number_of_edges()

    net_discription = 'BA_' + disc
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    os.makedirs(rand_net_dir, exist_ok=True)
    m = max([i for i in range(n_nodes // 2) if n_edges - (n_nodes - i) * i > 0]) - 1 # 27
    for net_id in range(start_n, start_n + n):
        print(net_id, f'Parameter m = {m}')
        graph = nx.extended_barabasi_albert_graph(n_nodes, m, 1 - (n_nodes-m)*m / n_edges, 0)
        align_edge_number(graph, n_edges)
        write_weights(graph, weights) # Write weights to the network
        w_r = nx.to_numpy_array(graph)
        node_index = np.arange(len(w_r))
        for _ in range(10): np.random.shuffle(node_index)
        w_r = w_r[node_index][:, node_index]
        matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
        np.savetxt(matrix_file, w_r, delimiter=',')

def generate_ws_networks(c, p, n = 100, shuffle = True, start_n = 0, connectivity='human'):
    ''' 
        Adjuest parameters to make the small-world property of the network similar to that of the BNA bfr matrix.
        And shuffle the node indices.
    '''
    connectome = Connectome(connectivity)
    raw_data, disc = connectome.get_connectivity(c)
    n_nodes = len(raw_data)
    weights = raw_data[np.triu_indices(n_nodes, 1)][raw_data[np.triu_indices(n_nodes, 1)].nonzero()]
    for _ in range(10): np.random.shuffle(weights)
    w = raw_data > 0
    raw_graph = nx.from_numpy_array(w)
    n_edges = raw_graph.number_of_edges()

    if shuffle:
        net_discription = 'WS_shuffle_' + disc
    else:
        net_discription = 'WS_' + disc
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    os.makedirs(rand_net_dir, exist_ok=True)
    k = round(n_edges / n_nodes) * 2
    for net_id in range(start_n, start_n + n):
        print(net_id, f'Parameter p = {p}; k = {k}')
        graph = nx.connected_watts_strogatz_graph(n_nodes, k, p)
        align_edge_number(graph, n_edges)
        write_weights(graph, weights) # Write weights to the network
        w_r = nx.to_numpy_array(graph)
        if shuffle:
            node_index = np.arange(len(w_r))
            for _ in range(10): np.random.shuffle(node_index)
            w_r = w_r[node_index][:, node_index]
        matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
        np.savetxt(matrix_file, w_r, delimiter=',')

if __name__ == '__main__':
    n = int(sys.argv[1])
    generate_randomize_networks(2, n, connectivity='BNA')
    generate_ws_networks_bna_bfr_sigma(n, True)
    generate_er_networks_bna_bfr(n)
    generate_ba_networks_bna_bfr(n)

    # generate_randomize_networks(5, n, connectivity='human')
    # generate_er_networks(5, n, connectivity='human')
    # generate_ba_networks(5, n, connectivity='human')
    # generate_ws_networks(5, p=0.34, n=n, connectivity='human')

    # generate_randomize_networks(0, n=n, connectivity='glasser')
    # generate_er_networks(0, n=n, connectivity='glasser')
    # generate_ba_networks(0, n=n, connectivity='glasser')
    # generate_ws_networks(0, p=0.38, n=n, connectivity='glasser')
    