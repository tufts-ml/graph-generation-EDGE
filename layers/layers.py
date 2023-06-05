import math
import numpy as np
import torch_geometric as pyg
import torch
import torch_scatter
from diffusion.diffusion_base import index_to_log_onehot
from torch.nn import functional as F
from torch import nn
from torch.nn.parameter import Parameter

norm_dict = {
    'Batch': lambda d: torch.nn.BatchNorm1d(d),
    'None': lambda d: torch.nn.Identity(),
    "Inst": lambda d: pyg.nn.norm.InstanceNorm(d),
    "Graph": lambda d: pyg.nn.norm.GraphNorm(d),
}

class SelEmb(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, t):
        out = self.act(t)
        out = self.linear(out)
        return out


class TimeEmb(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.act = act
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, t):
        out = self.act(t)
        out = self.linear(out)
        return out
        

class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class NodeModel(nn.Module):
    def __init__(self, num_bits, max_num_nodes, seq_lens, n_layers=6):
        super().__init__()
        self.num_bits = num_bits
        self.max_num_nodes = max_num_nodes
        self.seq_lens = seq_lens
        self.n_layers = n_layers
        self.embedding = nn.Linear(num_bits, 64)
        self.pos_embedding = SinusoidalPosEmb(64, seq_lens)
        self.g_embedding = nn.Linear(1, 256)
        self.res_embedding = nn.Linear(1, 64)
        self.lstm = nn.LSTM(input_size=64*3, hidden_size=256, num_layers=self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 128))
    def forward(self, x, g_v, res_count):
        x = self.embedding(x)
        g = g_v / self.max_num_nodes
        r = res_count / self.max_num_nodes

        g = self.g_embedding(g)
        r = self.res_embedding(r[..., None])
        t = torch.arange(0,x.shape[1])[None,:].repeat_interleave(x.shape[0], 0).to(x.device).view(-1)
        t = self.pos_embedding(t).view(x.shape[0],-1, 64)
        x = torch.cat([x, t, r], dim=-1)
        g = g[None,:,:].repeat_interleave(self.n_layers,0)
        x, _ = self.lstm(x, (g, g))
        x = self.linear(self.dropout(x))
        return x

class BitModel(nn.Module):
    def __init__(self, num_bits, max_num_nodes, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(3, 64)
        self.max_num_nodes = max_num_nodes
        self.n_layers = n_layers
        self.num_bits = num_bits
        self.pos_embedding = SinusoidalPosEmb(64, num_bits)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=self.n_layers, batch_first=True)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.res_embedding = nn.Linear(1, 128) 

    def forward(self, bits, hidden_nodes, res_count):
        x = self.embedding(bits)
        r = res_count/self.max_num_nodes
        r = self.res_embedding(r)
        t = torch.arange(0,x.shape[1])[None, :].repeat_interleave(x.shape[0], 0).to(x.device).view(-1)
        t = self.pos_embedding(t).view(x.shape[0], -1, 64)
        x = torch.cat([x, t], dim=-1)
        hidden_nodes = torch.cat([hidden_nodes, r],dim=-1)
        hidden_nodes = hidden_nodes[None,...].repeat_interleave(self.n_layers,0)
        x, _ = self.lstm(x, (hidden_nodes, hidden_nodes))
        x = self.linear(self.dropout(x))
        return x

class MiniAttentionLayer(torch.nn.Module):
    def __init__(self, node_dim, in_edge_dim, out_edge_dim, d_model, num_heads=2):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(d_model*num_heads, num_heads, batch_first=True)
        self.qkv_node = torch.nn.Linear(node_dim, d_model * 3 * num_heads)
        self.qkv_edge = torch.nn.Linear(in_edge_dim, d_model * 3 * num_heads)
        self.edge_linear = torch.nn.Sequential(torch.nn.Linear(d_model * num_heads, d_model), 
                                                torch.nn.SiLU(), 
                                                torch.nn.Linear(d_model, out_edge_dim))
    def forward(self, node_us, node_vs, edges):

        # node_us/vs: (B, D)
        q_node_us, k_node_us, v_node_us = self.qkv_node(node_us).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_node_vs, k_node_vs, v_node_vs = self.qkv_node(node_vs).chunk(3, -1) # (B, D*num_heads) for q/k/v
        q_edges, k_edges, v_edges = self.qkv_edge(edges).chunk(3, -1) # (B, D*num_heads) for q/k/v

        q = torch.stack([q_node_us, q_node_vs, q_edges], 1) # (B, 3, D*num_heads)
        k = torch.stack([k_node_us, k_node_vs, k_edges], 1) # (B, 3, D*num_heads)
        v = torch.stack([v_node_us, v_node_vs, v_edges], 1) # (B, 3, D*num_heads)

        h, _ = self.multihead_attn(q, k, v)
        h_edge = h[:, -1, :]
        h_edge = self.edge_linear(h_edge)

        return h_edge

class TGNN(torch.nn.Module):
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps, num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False, augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads 
        self.dim = dim
        self.num_steps = num_steps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        if 'gru' in kwargs.keys():
            if kwargs['gru']:
                self.gru = torch.nn.GRU(dim, dim)

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = pyg.nn.TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))
        self.node_interaction = MiniAttentionLayer(node_dim=dim, in_edge_dim=dim, out_edge_dim=dim, d_model=dim, num_heads=2)
        
        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )
        
    def forward(self, pyg_data, t_node, t_edge):
        edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
        is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]
        
        edge_index = pyg_data.full_edge_index[:, is_edge_indices]
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
        nodes = pyg.utils.degree(edge_index[0],num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree+1).long()

        nodes = nodes[..., None] / self.max_degree  # I prefer to make it embedding later
        nodes = self.embedding_t(nodes)
        t = self.time_pos_emb(t_node)
        t = self.mlp(t)
        
        h = nodes.unsqueeze(0)
        contexts = torch_scatter.scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)

        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            t_emb = self.layers[f'time{i}'](t)

            nodes = torch.cat([nodes, t_emb], dim=-1)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            ### gru update ###
            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            # aggregate locals to global
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = torch_scatter.scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)
            # spread global to locals
            nodes = nodes + contexts


        row, col = pyg_data.full_edge_index[0],  pyg_data.full_edge_index[1]
        edge_emb = torch.cat([nodes[row], nodes[col]], -1)
        edge_class = self.final_out(edge_emb)

        return pyg_data.log_node_attr, edge_class

class TGNN_degree_guided(torch.nn.Module):
    def __init__(self, max_degree, num_node_classes, num_edge_classes, dim, num_steps, num_heads=[4, 4, 4, 1], dropout=0., norm='None', degree=False, augmented_features={}, **kwargs) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.num_classes = num_edge_classes
        self.num_heads = num_heads 
        self.dim = dim
        self.num_steps = num_steps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.embedding_0 = torch.nn.Linear(1, dim)
        self.embedding_sel = torch.nn.Embedding(2, dim)
        self.node_in = torch.torch.nn.Sequential(
            torch.nn.Linear(dim * 3, dim),
            torch.nn.SiLU()
        )
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        if 'gru' in kwargs.keys():
            if kwargs['gru']:
                self.gru = torch.nn.GRU(dim, dim)

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = pyg.nn.TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = norm_dict[self.norm](dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))

        self.node_out_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*4, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim*2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim*2, dim*2)
        )
        
        self.final_out = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, self.num_classes)
        )
        
    def forward(self, pyg_data, t_node, t_edge):
        if hasattr(pyg_data, 'edge_index_t'):
            edge_index = pyg_data.edge_index_t

        else: 
            edge_attr_t = pyg_data.log_full_edge_attr_t.argmax(-1)
            is_edge_indices = edge_attr_t.nonzero(as_tuple=True)[0]

            edge_index = pyg_data.full_edge_index[:, is_edge_indices]
            edge_index = torch.cat([edge_index, edge_index.flip(0)],dim=-1)

        nodes_t = pyg.utils.degree(edge_index[0],num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree+1).long()
        node_selection = torch.zeros_like(nodes_t)


        nodes_t = nodes_t[..., None] / self.max_degree  # I prefer to make it embedding later
        nodes_0 = pyg_data.degree[..., None] / self.max_degree
        node_selection[pyg_data.active_node_indices] = 1
        node_selection = node_selection.long()
        
        nodes = torch.cat([self.embedding_t(nodes_t), self.embedding_0(nodes_0), self.embedding_sel(node_selection)], dim=-1)
        nodes = self.node_in(nodes)

        t = self.time_pos_emb(t_node)
        t = self.mlp(t)
        
        h = nodes.unsqueeze(0)
        contexts = torch_scatter.scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)

        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            t_emb = self.layers[f'time{i}'](t)

            nodes = torch.cat([nodes, t_emb], dim=-1)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            ### gru update ###
            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            # aggregate locals to global
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = torch_scatter.scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)
            # spread global to locals
            nodes = nodes + contexts

        # mlp add
        row = pyg_data.full_edge_index[0].index_select(0, pyg_data.active_edge_indices)
        col = pyg_data.full_edge_index[1].index_select(0, pyg_data.active_edge_indices)

        nodes = torch.cat([nodes, self.embedding_t(nodes_t), self.embedding_0(nodes_0), self.embedding_sel(node_selection)], dim=-1)
        nodes = self.node_out_mlp(nodes)

        edge_emb = nodes[row] + nodes[col]
        edge_class = self.final_out(edge_emb)
        
        return pyg_data.log_node_attr, edge_class