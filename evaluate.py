import torch
import pickle
import argparse
from diffusion.utils import add_parent_path
import pickle

import networkx as nx
import torch_geometric as pyg

# Data
add_parent_path(level=1)
from datasets.data import get_data

# Model
from model import get_model

###########
## Setup ##
###########
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='2023-05-29_18-29-35')
parser.add_argument('--dataset', type=str, default='polblogs')
parser.add_argument('--num_samples', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--checkpoint', type=int, default=5500)

eval_args = parser.parse_args()

torch.manual_seed(eval_args.seed)

log_dir = f'./wandb/{eval_args.dataset}/multinomial_diffusion/multistep/{eval_args.run_name}' 
path_args = '{}/args.pickle'.format(log_dir)
path_check = '{}/check/checkpoint_{}.pt'.format(log_dir, eval_args.checkpoint-1)

with open(path_args, 'rb') as f:
    args = pickle.load(f)

args.device = 'cuda:0'
train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics = get_data(args)

model = get_model(args, initial_graph_sampler=initial_graph_sampler)
checkpoint = torch.load(path_check, map_location=args.device)
model.load_state_dict(checkpoint['model'])

if torch.cuda.is_available():
    model = model.to(args.device)
model.eval()

# sample 
sampled_pygraph = model.sample(eval_args.num_samples)
pyg_datas = sampled_pygraph.to_data_list()
generated_nxgraphs = []

for pyg_data in pyg_datas:
    g_gen = pyg.utils.to_networkx(pyg_data, to_undirected=True)
    largest_cc = max(nx.connected_components(g_gen), key=len)
    g_gen = g_gen.subgraph(largest_cc)
    generated_nxgraphs.append(g_gen)  

# eval
print(test_evaluator.evaluate(generated_nxgraphs))