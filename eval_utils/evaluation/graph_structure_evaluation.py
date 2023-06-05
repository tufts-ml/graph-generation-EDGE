import networkx as nx
import dgl
import numpy as np
from scipy.linalg import toeplitz
import pyemd
import time
import concurrent.futures
from scipy.linalg import eigvalsh
import subprocess as sp
import os
from functools import partial
from sklearn.metrics.pairwise import pairwise_kernels
from eden.graph import vectorize

def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        return results, end - start
    return wrapper

class MMDEval():
    # Largely taken from the GraphRNN github: https://github.com/JiaxuanYou/graph-generation
    # I just rearranged to make it a little cleaner.
    def __init__(self, **kwargs):
        statistic = kwargs.get('statistic')
        if statistic == 'degree':
            self.descriptor = Degree(**kwargs)
        elif statistic == 'clustering':
            self.descriptor = Clustering(**kwargs)
        elif statistic == 'orbits':
            self.descriptor = Orbits(**kwargs)
        elif statistic == 'spectral':
            self.descriptor = Spectral(**kwargs)
        else:
            raise Exception('unsupported statistic'.format(statistic))


    def evaluate(self, generated_dataset=None, reference_dataset=None):
        # import ipdb; ipdb.set_trace()
        reference_dataset = self.extract_dataset(reference_dataset)
        generated_dataset = self.extract_dataset(generated_dataset)
        if len(reference_dataset) == 0 or len(generated_dataset) == 0:
            return {f'{self.descriptor.name}_mmd': 0}, 0

        start = time.time()
        metric = self.descriptor.evaluate(generated_dataset, reference_dataset)
        total = time.time() - start
        return {f'{self.descriptor.name}_mmd': metric}, total


    def extract_dataset(self, dataset):
        assert type(dataset) == list or type(dataset) == tuple, f'Unsupported type {type(dataset)} for \
                dataset, expected list of nx.Graph or dgl.DGLGraph'

        if isinstance(dataset[0], nx.Graph):
            pass
        elif isinstance(dataset[0], dgl.DGLGraph):
            dataset = [nx.Graph(g.cpu().to_networkx()) for g in dataset]
        else:
            raise Exception(f'Unsupported element type {type(dataset[0])} for dataset, \
                expected list of nx.Graph or dgl.DGLGraph')

        return [g for g in dataset if g.number_of_nodes() != 0]


class Descriptor():
    def __init__(self, is_parallel=True, bins=100, kernel='gaussian_emd', **kwargs):
        self.is_parallel = is_parallel
        self.bins = bins
        self.max_workers = kwargs.get('max_workers')

        if kernel == 'gaussian_emd':
            self.distance = self.emd
        # elif kernel == 'gaussian_tv':
        #     self.distance = self.gaussian_tv
        elif kernel == 'gaussian_rbf':
            self.distance = self.l2
            self.name += '_rbf'
        else:
            raise Exception(kernel)

        sigma_type = kwargs.get('sigma', 'single')
        if sigma_type == 'range':
            self.name += '_range'
            self.sigmas += [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
            self.__get_sigma_mult_factor = self.mean_pairwise_distance

        else:
            self.__get_sigma_mult_factor = self.identity

        self.sigmas = np.array(list(set(self.sigmas)))

    def get_sigmas(self, dists_GR):
        mult_factor = self.__get_sigma_mult_factor(dists_GR)
        return self.sigmas * mult_factor

    def mean_pairwise_distance(self, GR):
        return np.sqrt(GR.mean())

    def identity(self, *args, **kwargs):
        return 1

    def evaluate(self, generated_dataset, reference_dataset):
        ''' Compute the distance between the distributions of two unordered sets of graphs.
        Args:
          graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
        sample_pred = self.extract_features(generated_dataset)
        sample_ref = self.extract_features(reference_dataset)

        GG = self.disc(sample_pred, sample_pred, distance_scaling=self.distance_scaling)
        GR = self.disc(sample_pred, sample_ref, distance_scaling=self.distance_scaling)
        RR = self.disc(sample_ref, sample_ref, distance_scaling=self.distance_scaling)

        sigmas = self.get_sigmas(GR)
        max_mmd = 0
        for sigma in sigmas:
            gamma = 1 / (2 * sigma**2)

            K_GR = np.exp(-gamma * GR)
            K_GG = np.exp(-gamma * GG)
            K_RR = np.exp(-gamma * RR)

            mmd = K_GG.mean() + K_RR.mean() - (2 * K_GR.mean())
            max_mmd = mmd if mmd > max_mmd else max_mmd
            # print(mmd, max_mmd)

        return max_mmd

    def pad_histogram(self, x, y):
        support_size = max(len(x), len(y))
        # convert histogram values x and y to float, and make them equal len
        x = x.astype(np.float)
        y = y.astype(np.float)
        if len(x) < len(y):
            x = np.hstack((x, [0.0] * (support_size - len(x))))
        elif len(y) < len(x):
            y = np.hstack((y, [0.0] * (support_size - len(y))))

        return x, y


    def emd(self, x, y, distance_scaling=1.0):
        support_size = max(len(x), len(y))
        x, y = self.pad_histogram(x, y)

        d_mat = toeplitz(range(support_size)).astype(np.float)
        distance_mat = d_mat / distance_scaling

        dist = pyemd.emd(x, y, distance_mat)
        return dist ** 2

    def l2(self, x, y, **kwargs):
        x, y = self.pad_histogram(x, y)
        dist = np.linalg.norm(x - y, 2)
        return dist ** 2

    def gaussian_tv(self, x, y): #, sigma=1.0, *args, **kwargs):
        x, y = self.pad_histogram(x, y)

        dist = np.abs(x - y).sum() / 2.0
        return dist ** 2

    def kernel_parallel_unpacked(self, x, samples2, kernel):
        dist = []
        for s2 in samples2:
            dist += [kernel(x, s2)]
        return dist

    def kernel_parallel_worker(self, t):
        return self.kernel_parallel_unpacked(*t)

    def disc(self, samples1, samples2, **kwargs):
        ''' Discrepancy between 2 samples
        '''
        tot_dist = []
        if not self.is_parallel:
            for s1 in samples1:
                for s2 in samples2:
                    tot_dist += [self.distance(s1, s2)]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for dist in executor.map(self.kernel_parallel_worker,
                        [(s1, samples2, partial(self.distance, **kwargs)) for s1 in samples1]):
                    tot_dist += [dist]
        return np.array(tot_dist)

class Degree(Descriptor):
    def __init__(self, *args, **kwargs):
        self.name = 'degree'
        self.sigmas = [1.0]
        self.distance_scaling = 1.0
        super().__init__(*args, **kwargs)

    def extract_features(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for deg_hist in executor.map(self.degree_worker, dataset):
                    res.append(deg_hist)
        else:
            for g in dataset:
                degree_hist = self.degree_worker(g)
                res.append(degree_hist)

        res = [s1 / np.sum(s1) for s1 in res]
        return res

    def degree_worker(self, G):
        return np.array(nx.degree_histogram(G))

class Clustering(Descriptor):
    def __init__(self, *args, **kwargs):
        self.name = 'clustering'
        self.sigmas = [1.0 / 10]
        super().__init__(*args, **kwargs)
        self.distance_scaling = self.bins


    def extract_features(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for clustering_hist in executor.map(self.clustering_worker,
                    [(G, self.bins) for G in dataset]):
                    res.append(clustering_hist)
        else:
            for g in dataset:
                clustering_hist = self.clustering_worker((g, self.bins))
                res.append(clustering_hist)

        res = [s1 / np.sum(s1) for s1 in res]
        return res

    def clustering_worker(self, param):
        G, bins = param
        clustering_coeffs_list = list(nx.clustering(G).values())
        hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
        return hist

class Orbits(Descriptor):
    motif_to_indices = {
            '3path' : [1, 2],
            '4cycle' : [8],
    }
    COUNT_START_STR = 'orbit counts: \n'
    def __init__(self, *args, **kwargs):
        self.name = 'orbits'
        self.sigmas = [30.0]
        self.distance_scaling = 1
        super().__init__(*args, **kwargs)

    def extract_features(self, dataset):
        res = []
        for G in dataset:
            try:
                orbit_counts = self.orca(G)
            except:
                continue
            orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
            res.append(orbit_counts_graph)
        return np.array(res)

    def orca(self, graph):
        tmp_fname = f'./eval_utils/evaluation/orca/.tmp.txt'
        f = open(tmp_fname, 'w')
        f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
        for (u, v) in self.edge_list_reindexed(graph):
            f.write(str(u) + ' ' + str(v) + '\n')
        f.close()

        output = sp.check_output(['./eval_utils/evaluation/orca/orca', 'node', '4', tmp_fname, 'std'])
        output = output.decode('utf8').strip()

        idx = output.find(self.COUNT_START_STR) + len(self.COUNT_START_STR)
        output = output[idx:]
        node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ') ))
              for node_cnts in output.strip('\n').split('\n')])

        try:
            os.remove(tmp_fname)
        except OSError:
            pass

        return node_orbit_counts

    def edge_list_reindexed(self, G):
        idx = 0
        id2idx = dict()
        for u in G.nodes():
            id2idx[str(u)] = idx
            idx += 1

        edges = []
        for (u, v) in G.edges():
            edges.append((id2idx[str(u)], id2idx[str(v)]))
        return edges

class Spectral(Descriptor):
    def __init__(self, *args, **kwargs):
        self.name = 'spectral'
        self.sigmas = [1.0]
        self.distance_scaling = 1
        super().__init__(*args, **kwargs)

    def extract_features(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for spectral_density in executor.map(self.spectral_worker, dataset):
                    res.append(spectral_density)
        else:
            for g in dataset:
                spectral_temp = self.spectral_worker(g)
                res.append(spectral_temp)
        return res

    def spectral_worker(self, G):
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
        spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
        spectral_pmf = spectral_pmf / spectral_pmf.sum()
        return spectral_pmf


class NSPDKEvaluation():
    def evaluate(self, generated_dataset=None, reference_dataset=None):
        # prepare - dont include in timing
        generated_dataset_nx = [nx.Graph(g.cpu().to_networkx()) for g in generated_dataset if g.number_of_nodes() != 0]
        reference_dataset_nx = [nx.Graph(g.cpu().to_networkx()) for g in reference_dataset if g.number_of_nodes() != 0]

        if len(reference_dataset_nx) == 0 or len(generated_dataset_nx) == 0:
            return {'nspdk_mmd': 0}, 0

        if 'attr' not in generated_dataset[0].ndata:
            [nx.set_node_attributes(g, {key: str(val) for key, val in dict(g.degree()).items()}, 'label') for g in generated_dataset_nx]  # degree labels
            [nx.set_node_attributes(g, {key: str(val) for key, val in dict(g.degree()).items()}, 'label') for g in reference_dataset_nx]  # degree labels
            [nx.set_edge_attributes(g, '1', 'label') for g in generated_dataset_nx]  # degree labels
            [nx.set_edge_attributes(g, '1', 'label') for g in reference_dataset_nx]  # degree labels

        else:
            self.set_features(generated_dataset, generated_dataset_nx)
            self.set_features(reference_dataset, reference_dataset_nx)

        return self.evaluate_(generated_dataset_nx, reference_dataset_nx)

    def set_features(self, dset_dgl, dset_nx):
        for g_dgl, g_nx in zip(dset_dgl, dset_nx):
            feat_dict = {node: str(g_dgl.ndata['attr'][node].nonzero().item()) for node in range(g_dgl.number_of_nodes())}
            nx.set_node_attributes(g_nx, feat_dict, 'label')

            srcs, dests, eids = g_dgl.edges('all')
            feat_dict = {}
            for src, dest, eid in zip(srcs, dests, eids):
                feat_dict[(src.item(), dest.item())] = str(g_dgl.edata['attr'][eid].nonzero().item())
                # feat_dict = {edge: g.edata['attr'][edge].nonzero() for edge in range(g.number_of_edges())}
            # print(feat_dict)
            nx.set_edge_attributes(g_nx, feat_dict, 'label')

    @time_function
    def evaluate_(self, generated_dataset, reference_dataset):
        ref = vectorize(reference_dataset, complexity=4, discrete=True)
        for g in reference_dataset:
            del g

        gen = vectorize(generated_dataset, complexity=4, discrete=True)
        for g in generated_dataset:
            del g

        K_RR = pairwise_kernels(ref, ref, metric='linear', n_jobs=4)
        K_GG = pairwise_kernels(gen, gen, metric='linear', n_jobs=4)
        K_GR = pairwise_kernels(ref, gen, metric='linear', n_jobs=4)

        mmd = K_GG.mean() + K_RR.mean() - 2 * K_GR.mean()

        return {'nspdk_mmd': mmd}
