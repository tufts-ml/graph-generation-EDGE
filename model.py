import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion.diffusion_base import cosine_beta_schedule, linear_beta_schedule, Tt1_beta_schedule
from diffusion.diffusion_binomial_vanilla import BinomialDiffusionVanilla
from diffusion.diffusion_binomial_active import BinomialDiffusionActive
from layers.layers import *
from functools import partial

def add_model_args(parser):
    # Model params
    parser.add_argument('--loss_type', type=str, default='vb_kl')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_dim', type=int, default=64)
    parser.add_argument('--dp_rate', type=float, default=0.)
    parser.add_argument('--num_heads', type=int, nargs="*", default=[8, 8, 8, 8, 1])
    parser.add_argument('--final_prob_node', type=float, nargs="*", default=None)
    parser.add_argument('--final_prob_edge', type=float, nargs="*", default=[1-1e-12, 1e-12])
    parser.add_argument('--parametrization', type=str, default='x0')
    parser.add_argument('--sample_time_method', type=str, default='importance')
    parser.add_argument('--arch', type=str, help='GAT | TGNN')
    parser.add_argument('--noise_schedule', type=str, default='cosine', help='cosine | linear')
    parser.add_argument('--norm', type=str, default='None', help='None | BN' )
    
def get_model_id(args):
    return 'multinomial_diffusion'

def get_model(args, initial_graph_sampler):
    if args.final_prob_node is not None:
        assert sum(args.final_prob_node) == 1
        assert len(args.final_prob_node) == args.num_node_classes
    assert sum(args.final_prob_edge) == 1
    assert len(args.final_prob_edge) == args.num_edge_classes

    if args.arch == 'TGNN':
        assert args.parametrization in ('x0', 'xt')
        dynamics_fn = TGNN
        diffusion_fn = BinomialDiffusionVanilla
    elif args.arch == 'TGNN_degree_guided':
        assert args.parametrization in ('xt_prescribed_st')
        dynamics_fn = TGNN_degree_guided
        diffusion_fn = BinomialDiffusionActive
    else:
        raise NotImplementedError()
        
    dynamics = dynamics_fn(
            max_degree=args.max_degree,
            num_node_classes=2 if args.num_node_classes is None else args.num_node_classes, 
            num_edge_classes=args.num_node_classes,
            dim=args.diffusion_dim,
            num_steps=args.diffusion_steps,
            num_heads=args.num_heads,
            dropout=args.dp_rate,
            norm=args.norm,
            gru=True,
            degree=args.degree,
            augmented_features=args.augmented_feature_dict,
            return_node_class = args.has_node_feature
    )

    if args.noise_schedule == 'cosine':
        noise_schedule = cosine_beta_schedule
    elif args.noise_schedule == 'linear':
        noise_schedule = linear_beta_schedule
    elif args.noise_schedule == 'Tt1':
        noise_schedule = Tt1_beta_schedule
    else:
        raise NotImplementedError()

    base_dist = diffusion_fn(
        args.num_node_classes, args.num_edge_classes, initial_graph_sampler, dynamics, timesteps=args.diffusion_steps, 
        loss_type=args.loss_type, final_prob_node=args.final_prob_node, final_prob_edge=args.final_prob_edge,
        parametrization=args.parametrization, sample_time_method=args.sample_time_method,
        noise_schedule=noise_schedule, device=args.device)

    return base_dist
