import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from torch_scatter import scatter
import torch_geometric as pyg
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_sub_exp(a,b):
    assert torch.any(a > b), f'Error: {a > b}'
    return a + torch.log1p(-torch.exp(b-a))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def create_node_selections(log_x_t, log_x_tminus1, batched_graph):
    d_t = scatter(log_x_t.argmax(1), batched_graph.row, dim=1, dim_size=batched_graph.max_num_nodes) +  scatter(log_x_t.argmax(1), batched_graph.col, dim=1, dim_size=batched_graph.max_num_nodes) 
    d_tminus1 = scatter(log_x_tminus1.argmax(1), batched_graph.row, dim=1, dim_size=batched_graph.max_num_nodes) +  scatter(log_x_tminus1.argmax(1), batched_graph.col, dim=1, dim_size=batched_graph.max_num_nodes)
    return (d_tminus1 > d_t).long()


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return 1 - torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64).numpy()


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def Tt1_beta_schedule(timesteps):
    return 1/torch.linspace(1+1e-8, timesteps+1e-8, timesteps, dtype = torch.float64).flip(0).numpy() 


class DiffusionBase(torch.nn.Module):
    def __init__(self, num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps=1000,
                sample_time_method='importance', device='cuda'):
        super(DiffusionBase, self).__init__()

        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        
        self._denoise_fn = denoise_fn
        self.initial_graph_sampler = initial_graph_sampler
        self.num_timesteps = timesteps
        self.sample_time_method = sample_time_method
        self.device = device
        

    def _q_pred(self, batched_graph, t_node, t_edge):
        raise NotImplementedError()

    def _p_pred(self, batched_graph, t_node, t_edge):
        raise NotImplementedError()

    def _prepare_data_for_sampling(self, batched_graph):
        raise NotImplementedError()

    def _eval_loss(self, batched_graph):
        raise NotImplementedError()

    def _train_loss(self, batched_graph):
        raise NotImplementedError()

    def _sample_time(self, b, device, method):
        raise NotImplementedError()
    def _calc_num_entries(self, batched_graph):
        raise NotImplementedError()

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl


    def q_sample(self, batched_graph, t_node, t_edge):

        log_prob_node, log_prob_edge = self._q_pred(batched_graph, t_node, t_edge)

        # sample nodes
        log_out_node = self.log_sample_categorical(log_prob_node, self.num_node_classes)

        log_out_edge = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return log_out_node, log_out_edge 
    
    @torch.no_grad()
    def p_sample(self, batched_graph, t_node, t_edge):
        # p_sample is always one step prediction!
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph, t_node, t_edge)
        
        log_out_node = self.log_sample_categorical(log_model_prob_node, self.num_node_classes)

        log_out_edge = self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes)
        return log_out_node, log_out_edge

    def log_sample_categorical(self, logits, num_classes):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, num_classes)
        return log_sample


    def log_prob(self, batched_graph):
        if self.training:
            return self._train_loss(batched_graph)
        else:
            return self._eval_loss(batched_graph)


    def sample(self, num_samples):
        batched_graph = self.initial_graph_sampler.sample(num_samples)
        batched_graph.to(self.device)

        num_nodes = batched_graph.nodes_per_graph.sum()
        num_edges = batched_graph.edges_per_graph.sum()
        batched_graph = self._prepare_data_for_sampling(batched_graph)
 
        print()
        for t in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {t:4d}', end='\r')
            t_node = torch.full((num_nodes,), t, device=self.device, dtype=torch.long)
            t_edge = torch.full((num_edges,), t, device=self.device, dtype=torch.long)

            log_node_attr_tmin1, log_full_edge_attr_tmin1 = self.p_sample(batched_graph, t_node, t_edge)
            batched_graph.log_full_edge_attr_t = log_full_edge_attr_tmin1
            batched_graph.log_node_attr_t = log_node_attr_tmin1

        print()
        edge_attr = batched_graph.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr.nonzero(as_tuple=True)[0]

        edge_index = batched_graph.full_edge_index[:, is_edge_indices]
        batched_graph.edge_index = edge_index 

        edge_attr = edge_attr[is_edge_indices]
        batched_graph.edge_attr = edge_attr

        
        batched_graph.node_attr = batched_graph.log_node_attr_t.argmax(-1)

        # preparation for splitting batched graph
        # see https://github.com/pyg-team/pytorch_geometric/blob/259cfa7fb220d9cb504ab9de52bcd9dc5267befe/torch_geometric/data/separate.py#L12
        edge_slice = batched_graph.batch[batched_graph.edge_index[0]]
        edge_slice = scatter(torch.ones_like(edge_slice), edge_slice, dim_size=batched_graph.num_graphs )
        edge_slice = torch.nn.functional.pad(edge_slice, (1,0), 'constant', 0)
        edge_slice = torch.cumsum(edge_slice, 0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index'] = batched_graph._inc_dict['full_edge_index']

        return batched_graph
