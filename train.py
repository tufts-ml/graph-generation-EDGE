import numpy as np
import argparse
from diffusion.utils import add_parent_path, set_seeds

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Exp
from experiment import GraphExperiment, add_exp_args

# Model
from model import get_model, get_model_id, add_model_args

# Optim
from diffusion.optim.multistep import get_optim, get_optim_id, add_optim_args

###########
## Setup ##
###########


parser = argparse.ArgumentParser()
add_data_args(parser)
add_exp_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = parser.parse_args()
set_seeds(args.seed)

args.epochs = 50000
args.num_generation = 64
args.num_iter = 256
args.diffusion_dim = 64
args.diffusion_steps =512
args.device = 'cuda:0'
args.dataset = 'polblogs'
args.batch_size =4
args.clip_value =1
args.lr =1e-4
args.final_prob_edge = [1, 0]
args.sample_time_method = 'importance'
args.check_every = 50
args.eval_every = 50
args.noise_schedule = 'linear'
args.dp_rate =0.1
args.loss_type = 'vb_ce_xt_kl_st'
args.arch = 'TGNN_active'
args.parametrization = 'vb_ce_xt_kl_st'
args.degree = True
args.num_heads = [8, 8, 8, 8,1]


##################
## Specify data ##
##################

train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics = get_data(args)

args.num_edge_classes = num_edge_classes
args.num_node_classes = num_node_classes

if args.final_prob_node is None:
    args.final_prob_node = [1-1e-12, 1e-12]
    args.num_node_classes = 2
    args.has_node_feature = False

if 0 in args.final_prob_edge:
    args.final_prob_edge[np.argmax(args.final_prob_edge)] = args.final_prob_edge[np.argmax(args.final_prob_edge)]-1e-12
    args.final_prob_edge[np.argmin(args.final_prob_edge)] = 1e-12

args.max_degree = max_degree
args.num_node_feat = num_node_feat
args.augmented_feature_dict = augmented_feature_dict



data_id = get_data_id(args)
###################
## Specify model ##
###################

model = get_model(args, initial_graph_sampler=initial_graph_sampler)
model_id = get_model_id(args)
#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
optim_id = get_optim_id(args)

##############
## Training ##
##############
exp = GraphExperiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 eval_loader=eval_loader,
                 test_loader=test_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch,
                 monitoring_statistics=monitoring_statistics,
                 eval_evaluator=eval_evaluator, 
                 test_evaluator=test_evaluator,
                 n_patient=50)

exp.run()
