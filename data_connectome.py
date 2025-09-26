import numpy as np
import pandas as pd
import scipy.linalg
import scipy.io
import scipy.stats
import scipy.spatial
import networkx as nx
import bct
from enigmatoolbox.datasets import load_sc_as_one

from typing import Tuple, Union, Optional
from collections.abc import Sequence
from abc import ABC, abstractmethod
from functools import singledispatchmethod
import os

def _disc_c(net_discription, random_id, net_loop = 1000):
    if not isinstance(random_id, int):
        raise ValueError('Id must be an integer.')
    net_id = random_id % net_loop
    rand_net_dir = os.path.join('data', 'networks', net_discription)
    matrix_file = os.path.join(rand_net_dir, f'{net_discription}_{net_id}.csv')
    connectivity = np.loadtxt(matrix_file, delimiter = ',')
    return connectivity

def _shuffle_nodes(connectivity: np.ndarray) -> np.ndarray:
    ''' Randomly shuffle the nodes of the connectivity matrix.
    '''
    n = connectivity.shape[0]
    permutation = np.random.permutation(n)
    return connectivity[permutation][:, permutation]

class ConnMatrix(object):
    def _load_csv(self, filename: str) -> np.ndarray:
        matrix_file = os.path.join(self._file_path, filename)
        return np.loadtxt(matrix_file, delimiter = ',')
    
    def _load_npy(self, filename: str) -> np.ndarray:
        matrix_file = os.path.join(self._file_path, filename)
        return np.load(matrix_file).astype(float)
    
    def binarize(self, connectivity: np.ndarray) -> np.ndarray:
        connectivity[np.nonzero(connectivity)] = 1.
        return connectivity
    
    def positive(self, connectivity: np.ndarray) -> np.ndarray:
        connectivity[connectivity < 0] = 0
        return connectivity
    
    def negative(self, connectivity: np.ndarray) -> np.ndarray:
        connectivity[connectivity > 0] = 0
        return connectivity
    
    def absolute(self, connectivity: np.ndarray) -> np.ndarray:
        connectivity = np.abs(connectivity)
        return connectivity

    def threshold_1(self, connectivity: np.ndarray) -> np.ndarray:
        connectivity[np.abs(connectivity) < 1] = 0
        return connectivity
    
    def __call__(self, index, random_id = None):
        if index in self._matrices_map:
            fn, disc = self._matrices_map[index]
            return fn(random_id), disc
        else:
            raise NotImplementedError('No such connectivity matrix of BNA.')
        
    def __getitem__(self, index):
        if index in self._matrices_map:
            fn, disc = self._matrices_map[index]
            return fn, disc
        else:
            raise NotImplementedError('No such connectivity matrix of BNA.')

class BnaMatrix(ConnMatrix):
    # def _load_BNA_binary(self) -> np.ndarray:
    #     ''' connectivity = self._load_BNA_binary 相当于 connectivity, _ = self.get_connectivity(0) '''
    #     matrix_file = os.path.join('data', 'brainnetome', 'BNA_matrix_binary_246x246.csv')
    #     return np.loadtxt(matrix_file, delimiter = ',')
    def __init__(self) -> None:
        super().__init__()
        self._file_path = os.path.join('data', 'brainnetome')
        self._matrices_map = {
            0: (lambda _: None, ''),
            2: (lambda _: self._load_BNA_bfr(), 'BNA_matrix_bfr'),
            32: (lambda random_id: _disc_c('BNA_matrix_bfr_randmized', random_id), 'BNA_matrix_bfr_randmized'),
            51: (lambda random_id: _disc_c('WS_BNA_bfr_density_sigma_shuffle', random_id), 'WS_BNA_bfr_density_sigma_shuffle'),
            52: (lambda random_id: _disc_c('BA_BNA_bfr_density', random_id), 'BA_BNA_bfr_density'),
            53: (lambda random_id: _disc_c('ER_BNA_bfr_density', random_id), 'ER_BNA_bfr_density'),
        }
    
    def _load_BNA_bfr(self) -> np.ndarray:
        ''' connectivity = self._load_BNA_bfr 相当于 connectivity, _ = self.get_connectivity(1) '''
        matrix_file = os.path.join(self._file_path, 'matrix_signrank_bfr_avg.mat')
        return scipy.io.loadmat(matrix_file)['matrix_bfr_avg']

    def _load_BNA_fc(self) -> np.ndarray:
        ''' connectivity = self._load_BNA_fc 相当于 connectivity, _ = self.get_connectivity(2) '''
        matrix_file = os.path.join(self._file_path, 'FC_246.mat')
        return scipy.io.loadmat(matrix_file)['FC_mat_1']
    
    # def _load_BN_844(self) -> np.ndarray:
    #     matrix_file = os.path.join(self._file_path, 'BN_844sub_ave.txt')
    #     return np.loadtxt(matrix_file, delimiter = ' ')

    # def _load_BN_individual_binary(self, net_id: int) -> np.ndarray: # individual
    #     matrix_file = os.path.join(self._file_path, 'BN_1065ind_binary_75pct.npy')
    #     return np.load(matrix_file)[net_id].astype(float)
    
    def _load_BN_individual(self, net_id: int, f_name: str) -> np.ndarray: # individual
        matrix_file = os.path.join(self._file_path, f'{f_name}.npy')
        return np.load(matrix_file)[net_id].astype(float)
    
    
class HumanMatrix(ConnMatrix):
    def __init__(self, n='250'):
        super().__init__()
        self._file_path = os.path.join('data', 'human')
        # f_name = 'human_500' # 1015x1015
        # f_name = 'human_250' # 463x463
        f_name = f'human_{n}'
        self._matrices_map = {
            0: (lambda _: self._load_npy(f'{f_name}.npy'), f_name),
            1: (lambda random_id: _disc_c('Randomize_' + f_name, random_id), 'Randomize_' + f_name),
            2: (lambda random_id: _disc_c('ER_' + f_name, random_id), 'ER_' + f_name),
            3: (lambda random_id: _disc_c('WS_shuffle_' + f_name, random_id), 'WS_shuffle_' + f_name), # p=0.8
            4: (lambda random_id: _disc_c('BA_' + f_name, random_id), 'BA_' + f_name),
        }
        
class EnigmaMatrix(ConnMatrix):
    def __init__(self, parcellation='aparc'):
        super().__init__()
        self._file_path = os.path.join('data', 'enigma')
        f_name = parcellation
        self._matrices_map = {
            0: (lambda _: self.load_enigma_matrix(parcellation), f_name),
            1: (lambda random_id: _disc_c('Randomize_' + f_name, random_id), 'Randomize_' + f_name),
            2: (lambda random_id: _disc_c('ER_' + f_name, random_id), 'ER_' + f_name),
            3: (lambda random_id: _disc_c('WS_shuffle_' + f_name, random_id), 'WS_shuffle_' + f_name), # aparc p=0.3; schaefer_400 p=0.39 ; glasser_360 p=0.38
            4: (lambda random_id: _disc_c('BA_' + f_name, random_id), 'BA_' + f_name),
        }
            
    def load_enigma_matrix(self, parcellation):
        return load_sc_as_one(parcellation)[0]
        
class Connectome(object):
    def __init__(self, connectivity: str = 'BNA') -> None:
        self.connectivity = connectivity
        if connectivity == 'BNA':
            self._load_subregion()
            self._repr = 'Brannetome Atlas'
            self._matrices = BnaMatrix()
        elif connectivity == 'human':
            self._repr = 'human'
            self._matrices = HumanMatrix()
        elif connectivity == 'aparc':
            self._repr = 'aparc'
            self._matrices = EnigmaMatrix('aparc')
        elif connectivity == 'schaefer':
            self._repr = 'schaefer'
            self._matrices = EnigmaMatrix('schaefer_400')
        elif connectivity == 'glasser':
            self._repr = 'glasser'
            self._matrices = EnigmaMatrix('glasser_360')
            
        else:
            raise NotImplementedError()

    def __repr__(self) -> str:
        return self._repr
    
    def __call__(self, c: int, random_id: Optional[int] = None) -> Tuple[np.ndarray, str]:
        return self.get_connectivity(c, random_id)
    
    @staticmethod
    def set_connectivity_matrix(c: np.ndarray, alpha: float = 1.0, normalize: bool = False, symmetric: bool = False, ):
        if symmetric:
            c = Connectome.symmetric_connectivity(c)
        if normalize:
            c = Connectome.normalize_connectivity(c)
        c = Connectome.set_spectrum_radius(c, alpha)
        return c
    
    def wighted_eucidean_distance(self, c: Union[int, str], random_id: Optional[int] = None) -> float:
        w, _ = self.get_connectivity(c, random_id)
        w = self.set_connectivity_matrix(w)
        dist = self.distance_matrix
        return (w * dist).sum()
    
    @property
    def n_subregions(self):
        return len(self.subregions)
    @property
    def mni_coordinates(self):
        return np.asarray([i for i in self.subregions['MNI(X,Y,Z)']])
    @property
    def distance_matrix(self, metric='euclidean'):
        ''' Calculate the distance matrix between nodes based on MNI coordinates. '''
        coor = np.asarray([i for i in self.subregions['MNI(X,Y,Z)']])
        dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coor, metric=metric))
        return dist
    @property
    def hemisphere_id(self, right=0, left=1):
        return self.subregions['Hemisphere'].map({'right': right, 'left': left}).to_numpy()

    def _load_subregion(self):
        '''
            Load the subregion information of all nodes.
            Load path:
            'data', 'brainnetome', 'BNA_subregions.xlsx' / 'subregion_func_network_Yeo_updated.csv'
            Add attributes:
            self.subregions: DataFrame of anatomical partition information for each node
            self.lobes: List of brain partition names
            self.subregion_func_network_Yeo: DataFrame of functional partition information for each node
            self.yeo_7network: List of 7 network names
            self.yeo_17network: List of 17 network names
        '''
        filename = os.path.join('data', 'brainnetome', 'BNA_subregions.xlsx')
        df = pd.read_excel(filename)
        lobe = []
        gyrus = []
        for i in range(len(df)):
            if pd.notna(df.loc[i, 'Lobe']):
                tmp = df.loc[i, 'Lobe'].strip() # have to remove the space
                lobe.append(tmp)
            df.loc[i, 'Lobe'] = tmp
        for i in range(len(df)):
            if pd.notna(df.loc[i, 'Gyrus']):
                tmp = df.loc[i, 'Gyrus'].strip() # have to remove the space
                gyrus.append(tmp)
            df.loc[i, 'Gyrus'] = tmp
        lobe = tuple(lobe)
        gyrus = tuple(gyrus)
        
        df.rename(columns={'Unnamed: 5': 'Anatomical and modified Cyto-architectonic descriptions', 'Anatomical and modified Cyto-architectonic descriptions': 'Unnamed: 5'}, inplace=True)
        df.loc[:, 'Anatomical and modified Cyto-architectonic descriptions'] = df['Anatomical and modified Cyto-architectonic descriptions'].apply(str.strip)
        # df.loc[:, 'Left and Right Hemisphere'] = df['Left and Right Hemisphere'].apply(str.strip)
        df.drop('Left and Right Hemisphere', axis=1, inplace=True)
        df.drop('Unnamed: 5', axis=1, inplace=True)
        df1 = df.set_index('Label ID.L')
        df1.drop(['rh.MNI(X,Y,Z)', 'Label ID.R'], axis=1, inplace=True)
        df1['Hemisphere'] = 'left'
        df2 = df.set_index('Label ID.R')
        df2.drop(['lh.MNI(X,Y,Z)', 'Label ID.L'], axis=1, inplace=True)
        df2['Hemisphere'] = 'right'
        df1.rename(columns={'lh.MNI(X,Y,Z)': 'MNI(X,Y,Z)'}, inplace=True)
        df2.rename(columns={'rh.MNI(X,Y,Z)': 'MNI(X,Y,Z)'}, inplace=True)
        subregions = pd.concat([df1, df2]).sort_index()
        subregions.loc[:, 'MNI(X,Y,Z)'] = subregions['MNI(X,Y,Z)'].apply(lambda x: tuple(int(i) for i in x.split(',')))

        network_file = os.path.join('data', 'brainnetome', 'subregion_func_network_Yeo_updated.csv')
        df = pd.read_csv(network_file)
        df1 = df.iloc[2:9, 10:]
        df2 = df.iloc[12:29, 10:]
        yeo7 = dict(zip(map(int, df1.iloc[:, 0]), df1.iloc[:, 1]))
        yeo17 = dict(zip(map(int, df2.iloc[:, 0]), df2.iloc[:, 1]))
        yeo7[0] = ''
        yeo17[0] = ''
        df = df.iloc[:, :5]
        df.loc[:, 'Yeo 7 Network name'] = df['Yeo_7network'].apply(yeo7.get)
        df.loc[:, 'Yeo 17 Network name'] = df['Yeo_17network'].apply(yeo17.get)
        df = df.set_index('Label')
        df.loc[:, 'subregion_name'] = df['subregion_name'].apply(str.strip)
        df.loc[:, 'region'] = df['region'].apply(str.strip)
        df.rename(columns={'subregion_name': 'Subregion name', 'region': 'Region', 'Yeo_7network': 'Yeo 7 Network', 'Yeo_17network': 'Yeo 17 Network'}, inplace=True)
        df.insert(0, 'Lobe', subregions['Lobe'])
        df.insert(1, 'Gyrus', subregions['Gyrus'])
        df.insert(4, 'Anatomical and modified Cyto-architectonic descriptions', subregions['Anatomical and modified Cyto-architectonic descriptions'])
        df.insert(5, 'MNI(X,Y,Z)', subregions['MNI(X,Y,Z)'])
        df.insert(5, 'Hemisphere', subregions['Hemisphere'])
        subregions = df
        
        self.subregions = subregions
        self.subregion_func_network_Yeo = subregions

        self.yeo_7network = tuple(yeo7[i] for i in range(len(yeo7)))
        self.yeo_17network = tuple(yeo17[i] for i in range(len(yeo17)))
        self.lobes = lobe
        self.gyri = gyrus

    def node_community(self, c: str = 'Lobe', out: str = 'array'):
        c = c.lower()
        if c == 'lobe':
            d = dict(zip(self.lobes, range(len(self.lobes))))
            community = self.subregions['Lobe'].apply(lambda x: d[x]).to_numpy()
        elif c == 'yeo 7 network':
            d = dict(zip(self.yeo_7network, range(len(self.yeo_7network))))
            community = self.subregion_func_network_Yeo['Yeo 7 Network name'].apply(lambda x: d[x]).to_numpy()
        elif c == 'yeo 17 network':
            d = dict(zip(self.yeo_17network, range(len(self.yeo_17network))))
            community = self.subregion_func_network_Yeo['Yeo 17 Network name'].apply(lambda x: d[x]).to_numpy()
        elif c == 'gyrus':
            d = dict(zip(self.gyri, range(len(self.gyri))))
            community = self.subregions['Gyrus'].apply(lambda x: d[x]).to_numpy()
        else:
            raise ValueError('Region not found.')
        if out == 'array':
            return community
        elif out == 'set':
            subregions = np.unique(community)
            return [set(np.where(community == i)[0]) for i in subregions]
        else:
            raise NotImplementedError

    def regions_to_labels(self, regions: Union[str, Sequence[str]], logical_not: bool = False, rotation_id: Optional[int] = None):
        '''
            Returns a numpy boolean array with the same length as the number of nodes,
        '''
        regions = [regions] if isinstance(regions, str) else regions
        if regions is None:
            labels = np.array()
        elif np.array([i in self.lobes for i in regions]).all():
            labels = self.subregions['Lobe'].apply(lambda x: x in regions).to_numpy()
        elif np.array([i in self.yeo_7network for i in regions]).all():
            labels = self.subregion_func_network_Yeo['Yeo 7 Network name'].apply(lambda x: x in regions).to_numpy()
        elif np.array([i in self.yeo_17network for i in regions]).all():
            labels = self.subregion_func_network_Yeo['Yeo 17 Network name'].apply(lambda x: x in regions).to_numpy()
        elif np.array([i in self.gyri for i in regions]).all():
            labels = self.subregions['Gyrus'].apply(lambda x: x in regions).to_numpy()
        else:
            raise ValueError('Region not found.')
        if rotation_id != None:
            i = rotation_id % 1000
            rotation_file = os.path.join('data', 'networks', 'BNA_node_rotation', 'BNA_node_rotation_%d.csv'%(i))
            rotations = np.loadtxt(rotation_file, dtype = int, delimiter = ',')
            labels = labels[rotations]
        if logical_not:
            return ~labels
        else:
            return labels
        
    def lobe_numbers(self, ):
        return np.array([sum(self.regions_to_labels(i)) for i in self.lobes])
    def network_numbers(self, ):
        return np.array([sum(self.regions_to_labels(i)) for i in self.yeo_7network if i])
    
    def get_labels(self, input_region: Sequence[str] = [], output_region: Sequence[str] = []):
        '''
            Input two lists of brain regions, which represent the range of all 
            brain regions corresponding to input and output nodes respectively.
            Return a numpy boolean array with the same length as the number of 
            nodes, indicating whether each node belongs to the union of these regions.
            If you only want to input one list, you can call `regions_to_labels`.
        '''
        if self._repr == 'Brannetome Atlas':
            if not input_region:
                input_region = ['Subcortical Nuclei']
            if not output_region:
                output_region = ['Frontal Lobe', 'Temporal Lobe', 'Parietal Lobe', 'Occipital Lobe', 'Insular Lobe', 'Limbic Lobe']
            input_labels = self.regions_to_labels(input_region)
            output_labels = self.regions_to_labels(output_region)
        elif self._repr == 'human':
            cortical_file =  os.path.join('data', 'human', f'cortical_human_250.npy')
            cortical_labels = np.load(cortical_file).astype(bool)
            input_labels, output_labels = ~cortical_labels, cortical_labels
        elif self._repr in ['aparc', 'schaefer', 'glasser']:
            n = self.conn(0).shape[0]
            input_labels, output_labels = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
            input_labels[n-14:] = True
            output_labels[:n-14] = True
        else:
            raise NotImplementedError()
        return input_labels, output_labels

    def conn(self, c: Union[int, str], random_id: Optional[int] = None) -> np.ndarray:
        conn, _ = self._matrices(c, random_id)
        return conn
    def conn_name(self, c: Union[int, str], ) -> str:
        _, disc = self._matrices[c]
        return disc
    def get_connectivity(self, c: int, random_id: Optional[int] = None) -> Tuple[np.ndarray, str]:
        '''
            Input an integer to indicate the type of connectivity matrix.
            Returns a connectivity matrix and a string describing the connectivity matrix.
        '''
        return self._matrices(c, random_id)

    @staticmethod
    def normalize_connectivity(connectivity: np.ndarray):
        ''' 
            Normalize the connectivity matrix to the range [0, 1].
        '''
        un_connected_edges = np.where(connectivity == 0)
        connectivity = (connectivity - connectivity.min()) / (connectivity.max() - connectivity.min())
        connectivity[un_connected_edges] = 0
        return connectivity
    @staticmethod
    def symmetric_connectivity(connectivity: np.ndarray):
        return (connectivity + connectivity.T) / 2.
    @staticmethod
    def calculate_spectrum_radius(connectivity):
        eigenvalues = scipy.linalg.eigvals(connectivity) # Eigenvalues of the matrix
        spectrum_radius = np.max(np.abs(eigenvalues)) # If the matrix is non-negative, the spectral radius is one of the eigenvalues.
        return spectrum_radius
    @staticmethod
    def set_spectrum_radius(connectivity, alpha = 1.):
        '''
            Set the spectrum radius of connectivity matrix to alpha.
        '''
        spectrum_radius = Connectome.calculate_spectrum_radius(connectivity)
        return connectivity / spectrum_radius * alpha