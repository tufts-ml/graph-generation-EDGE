import math
import torch
import os 
import networkx as nx
import numpy as np

import pickle as pkl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch_geometric.data import data
import torch_geometric as pyg
import random
from functools import partial
from torch_geometric.datasets import QM9
from datasets.data_utils import EmpiricalEmptyGraphGenerator, NeuralEmptyGraphGenerator, preprocess, collate_fn, FEATURE_EXTRACTOR
from datasets.evaluator import NetworkEvaluator, GenericGraphEvaluator


class NetworkDataset(Dataset):
    def __init__(self, pyg_graph, num_iter, transform=None):
        super().__init__()
        self.pyg_data = pyg_graph
        self.transform = transform
        self.num_iter = num_iter

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.pyg_graph)
        return self.pyg_data

    def __len__(self):
        return self.num_iter


class GraphDataset(Dataset):
    def __init__(self, pyg_datas):
        super().__init__()
        self.pyg_datas = pyg_datas

    def __getitem__(self, index):
        return self.pyg_datas[index]#, self.denses[index]

    def __len__(self):
        return len(self.pyg_datas)


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str)
    # Train params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_iter', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=eval, default=True)

    parser.add_argument('--empty_graph_sampler', type=str, default='empirical', help='empirical | neural') 
    parser.add_argument('--degree', action='store_true')
    parser.add_argument('--augmented_features', type=str, nargs="*", default=[])

def get_data_id(args):
    return '{}'.format(args.dataset)

def get_data(args):
    if args.dataset in ['cora', 'polblogs', 'Homo_sapiens']:
        repeat = 1
        num_node_classes = None
        num_edge_classes = 2
        num_node_feat = None
        nx_graph = pkl.load(open(f'graphs/{args.dataset}.pkl','rb'))
        pyg_graph = preprocess(nx_graph, degree=args.degree)
        max_degree = max([d for _, d in nx_graph.degree()]) 
        train_set = NetworkDataset(pyg_graph, num_iter=args.num_iter * args.batch_size, transform=None)
        test_set = eval_set = NetworkDataset(pyg_graph, num_iter=100, transform=None)
        initial_graph_sampler = EmpiricalEmptyGraphGenerator([train_set[0]], degree=args.degree, augment_features=args.augmented_features)
        eval_evaluator = test_evaluator = NetworkEvaluator(nx_graph)
        monitoring_statistics = ['nmae/assortativity','nmae/triangle_count', 'nmae/clustering_coefficient']
 
    elif args.dataset in ['community',  'Ego']:
        repeat = 64 
        num_node_classes = None
        num_edge_classes = 2
        num_node_feat = None
        nx_graphs = pkl.load(open(f"graphs/{args.dataset}.pkl", 'rb'))
        random.shuffle(nx_graphs)
        l = len(nx_graphs)
        train_nx_graphs = nx_graphs[:int(0.8*l)]
        eval_nx_graphs = nx_graphs[:int(0.2*l)]
        test_nx_graphs = nx_graphs[int(0.8*l):] 

        train_pygraphs = []
        eval_pygraphs = []
        test_pygraphs = []

        max_degree = max([max([d for n, d in train_nx_graph.degree()]) for train_nx_graph in train_nx_graphs])
        for nx_graph in train_nx_graphs:
            pyg_data = preprocess(nx_graph, degree=args.degree)
            train_pygraphs.append(pyg_data)

        for nx_graph in eval_nx_graphs:
            pyg_data = preprocess(nx_graph, degree=args.degree)
            eval_pygraphs.append(pyg_data)

        for nx_graph in test_nx_graphs:
            pyg_data = preprocess(nx_graph, degree=args.degree)
            test_pygraphs.append(pyg_data)
            
        train_set = ConcatDataset([GraphDataset(train_pygraphs) for _ in range(repeat)])
        eval_set = GraphDataset(eval_pygraphs)
        test_set = GraphDataset(test_pygraphs)

        if args.empty_graph_sampler == 'empirical':
            initial_graph_sampler = EmpiricalEmptyGraphGenerator(train_pygraphs, degree=args.degree)
        elif args.empty_graph_sampler == 'neural':
            neural_attr_sampler = torch.load(f'graphs/{args.dataset}_degree_sampler.pt', map_location=args.device)
            initial_graph_sampler = NeuralEmptyGraphGenerator(train_pygraphs, neural_attr_sampler, degree=args.degree, device=args.device)

        eval_evaluator = GenericGraphEvaluator(eval_nx_graphs, device=args.device)
        test_evaluator = GenericGraphEvaluator(test_nx_graphs, device=args.device)

        monitoring_statistics = ['clustering_mmd', 'orbits_mmd', 'spectral_mmd', 'degree_mmd', 'mmd_linear', 'mmd_rbf']

    else:
        raise NotImplementedError
    augmented_feature_dict = {k:FEATURE_EXTRACTOR[k]['data_spec'] for k in args.augmented_features}

    # Data Loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size*repeat, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=partial(collate_fn))
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)

    return train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics
 
