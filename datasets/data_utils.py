import torch
import random
import networkx as nx
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj
from torch_geometric import transforms as T
from torch.nn import functional as F
from layers.layers import BitModel, NodeModel

FEATURE_EXTRACTOR = {
}


def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    
def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def unpack_deg_matrix(degs):
    res = []
    for deg in degs:
        deg = deg.long().tolist()
        r = []
        for d in deg:
            if (sum(r)==0) or (d > 0):
                r.append(d)
        res.append(r)
    return res


def deg_hist_to_deg_seq(deg_hist):
    ret = torch.zeros(sum(deg_hist))
    cum = 0
    for d, num_nodes in enumerate(deg_hist):
        ret[cum:cum+num_nodes] = d+1
        cum = cum+num_nodes
    return ret

def preprocess(g, degree=False, augmented_features=[]):
    if isinstance(g, nx.Graph):
        pyg_data = pyg.utils.from_networkx(g)
        adj = torch.from_numpy(nx.to_numpy_array(g).astype(np.int)).long()
    elif isinstance(g, pyg.data.Data):
        pyg_data = g
        adj = to_dense_adj(g.edge_index)[0].long()
    else:
        raise NotImplementedError()
        
    row, col = torch.triu_indices(pyg_data.num_nodes, pyg_data.num_nodes,1)
    pyg_data.full_edge_index = torch.stack([row, col])

    pyg_data.full_edge_attr = adj[pyg_data.full_edge_index[0], pyg_data.full_edge_index[1]]

    if not hasattr(pyg_data, 'node_attr'):
        pyg_data.node_attr = torch.zeros(pyg_data.num_nodes, dtype=torch.long)

    if degree:
        pyg_data.degree = pyg.utils.degree(pyg_data.edge_index[0]).long() # make sure edge_index is bi-directional
    
    for augmented_feature in augmented_features:
        setattr(pyg_data, augmented_feature, FEATURE_EXTRACTOR[augmented_feature]['func'](pyg_data))

    return pyg_data

def collate_fn(pyg_datas, repeat=1):
    pyg_datas = sum([[pyg_data.clone() for _ in range(repeat)]for pyg_data in pyg_datas],[])
    batched_data = pyg.data.Batch.from_data_list(pyg_datas)
    batched_data.nodes_per_graph = torch.tensor([pyg_data.num_nodes for pyg_data in pyg_datas])
    batched_data.edges_per_graph = torch.tensor([pyg_data.num_nodes * (pyg_data.num_nodes-1)//2 for pyg_data in pyg_datas])

    return batched_data 


class EmpiricalEmptyGraphGenerator:
    def __init__(self, train_pyg_datas, degree=False, augment_features=[]):
        # pmf of graph size
        num_nodes = torch.tensor([pyg_data.num_nodes for pyg_data in train_pyg_datas])

        self.min_node = num_nodes.min().long().item()
        self.max_node = num_nodes.max().long().item()

        unnorm_p = torch.histc(num_nodes.float(), bins=self.max_node-self.min_node+1)

        self.empirical_graph_size_dist = unnorm_p/unnorm_p.sum()

        # empty graph table
        self.empty_graphs = {}

        # degree table
        self.degree = degree
        self.augment_features = augment_features

        self.empirical_node_feat_dist = {}

        for pyg_data in train_pyg_datas:
            if pyg_data.num_nodes not in self.empirical_node_feat_dist:
                self.empirical_node_feat_dist[pyg_data.num_nodes] = []
            feats = {}
            if self.degree:
                feats['degree'] = pyg.utils.degree(pyg_data.edge_index[0],num_nodes=pyg_data.num_nodes)
            for feat_name in self.augment_features:
                feats[feat_name] = getattr(pyg_data, feat_name)# FEATURE_EXTRACTOR[feat_name]['func'](pyg_data)
            # feats['x'] = pyg_data.x
            self.empirical_node_feat_dist[pyg_data.num_nodes].append(feats)


    def _sample_graph_size_and_features(self, num_samples):
        ret = self.empirical_graph_size_dist.multinomial(num_samples=num_samples, replacement=True) + self.min_node
        ret = ret.tolist()
        xT_feats = [] 
        for n_node in ret:
            xT_feats.append(random.choice(self.empirical_node_feat_dist[n_node]))
        # xT_feats will be a list of dicts
        return ret, xT_feats

    def _generate_empty_data(self, num_node_per_graphs, xT_feats):
        return_data_list = []

        for num_node, xT_feat in zip(num_node_per_graphs, xT_feats):
            if num_node not in self.empty_graphs:
                pyg_data = pyg.data.Data()
                row, col = torch.triu_indices(num_node, num_node,1)
                pyg_data.full_edge_index = torch.stack([row, col])

                pyg_data.full_edge_attr = torch.zeros((pyg_data.full_edge_index[0].shape[0],), dtype=torch.long)
                pyg_data.node_attr = torch.zeros((num_node,), dtype=torch.long)

                pyg_data.num_nodes = num_node
                self.empty_graphs[num_node] = pyg_data

            pyg_data = self.empty_graphs[num_node].clone()
            for feat_name in xT_feat:
                setattr(pyg_data, feat_name, xT_feat[feat_name])
            
            return_data_list.append(pyg_data)

        batched_data = collate_fn(return_data_list)
        return batched_data

    def sample(self, num_samples):
        num_node_per_graphs, xT_feats = self._sample_graph_size_and_features(num_samples)
        empty_pyg_datas = self._generate_empty_data(num_node_per_graphs, xT_feats)
        return empty_pyg_datas

class NeuralEmptyGraphGenerator:
    def __init__(self, train_pyg_datas, neural_attr_sampler, degree=False, device='cuda:0'):
        # now only support degree features, other features are left to future.

        num_nodes = torch.tensor([pyg_data.num_nodes for pyg_data in train_pyg_datas])

        self.min_node = num_nodes.min().long().item()
        self.max_node = num_nodes.max().long().item()

        unnorm_p = torch.histc(num_nodes.float(), bins=self.max_node-self.min_node+1)
        # empty graph table
        self.empty_graphs = {}
        self.empirical_graph_size_dist = unnorm_p/unnorm_p.sum()
        self.degree = degree
        self.neural_attr_sampler = neural_attr_sampler
        self.device = device
        
        self.node_model = NodeModel(num_bits=neural_attr_sampler['NUM_BITS'], max_num_nodes=neural_attr_sampler['MAX_NUM_NODES'], seq_lens=neural_attr_sampler['SEQ_LENS'])
        self.bit_model = BitModel(num_bits=neural_attr_sampler['NUM_BITS'], max_num_nodes=neural_attr_sampler['MAX_NUM_NODES'])
        
        self.node_model.to(self.device)
        self.bit_model.to(self.device)

        self.node_model.load_state_dict(neural_attr_sampler['modelNode'])
        self.bit_model.load_state_dict(neural_attr_sampler['modelBit'])
 
    def _sample_graph_size_and_features(self, num_samples):
        ret = self.empirical_graph_size_dist.multinomial(num_samples=num_samples, replacement=True) + self.min_node
        if self.degree:
            x = torch.zeros(num_samples, 1, self.neural_attr_sampler['NUM_BITS'])
            g = r = ret[:, None]
            x = x.to(self.device)
            g = g.to(self.device)
            r = r.to(self.device)
            self.node_model.eval()
            self.bit_model.eval()
            with torch.no_grad():
                for i in range(self.neural_attr_sampler['SEQ_LENS']):      
                    node_hidden = self.node_model(x, g, r)[:,-1,:]
                    y = (torch.ones(num_samples, 1).long().to(self.device)*2).long()

                    for j in range(self.neural_attr_sampler['NUM_BITS']):
                        prediction = self.bit_model(y.view(-1, j+1), node_hidden.view(-1, node_hidden.shape[-1]), r[:,-1][:,None])[:,-1,:]
                        prediction = F.sigmoid(prediction)
                        index = prediction.bernoulli().long()
                        y = torch.cat([y, index],dim=-1)
                    y = y[:, 1:]
                    n_j = bin2dec(y, self.neural_attr_sampler['NUM_BITS'])-1
                    r = torch.cat([r, (r[:, -1]-n_j)[:,None]],dim=-1)
                    x = torch.cat([x, y[:,None,:]], dim=1)
                
                x = (bin2dec(x, self.neural_attr_sampler['NUM_BITS'])-1).clamp(0)[:, 1:]
            xT_feats = unpack_deg_matrix(x)
            ret = [sum(xT_feat) for xT_feat in xT_feats]
            xT_feats = [{'degree':deg_hist_to_deg_seq(xT_feat)} for xT_feat in xT_feats]
        else:
            xT_feats = [{} for _ in ret]
            ret = ret.tolist()
        return ret, xT_feats

    def _generate_empty_data(self, num_node_per_graphs, xT_feats):
        return_data_list = []

        for num_node, xT_feat in zip(num_node_per_graphs, xT_feats):
            if num_node not in self.empty_graphs:
                pyg_data = pyg.data.Data()
                row, col = torch.triu_indices(num_node, num_node,1)
                pyg_data.full_edge_index = torch.stack([row, col])

                pyg_data.full_edge_attr = torch.zeros((pyg_data.full_edge_index[0].shape[0],), dtype=torch.long)
                pyg_data.node_attr = torch.zeros((num_node,), dtype=torch.long)

                pyg_data.num_nodes = num_node
                self.empty_graphs[num_node] = pyg_data

            pyg_data = self.empty_graphs[num_node].clone()
            for feat_name in xT_feat:
                setattr(pyg_data, feat_name, xT_feat[feat_name])
            
            return_data_list.append(pyg_data)

        batched_data = collate_fn(return_data_list)
        return batched_data

    def sample(self, num_samples):
        num_node_per_graphs, xT_feats = self._sample_graph_size_and_features(num_samples)
        empty_pyg_datas = self._generate_empty_data(num_node_per_graphs, xT_feats)
        return empty_pyg_datas

