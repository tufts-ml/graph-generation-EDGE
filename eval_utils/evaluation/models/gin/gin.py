"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
# from dgl.nn.pytorch.conv import GINConv
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

class GINConv(nn.Module):
    r"""
    Description
    -----------
    Graph Isomorphism Network layer from paper `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__.
    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)
    If a weight tensor on each edge is provided, the weighted graph convolution is defined as:
    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{e_{ji} h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)
    where :math:`e_{ji}` is the weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that `e_{ji}` is broadcastable with `h_j^{l}`.
    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.
    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GINConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> lin = th.nn.Linear(10, 10)
    >>> conv = GINConv(lin, 'max')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.1804,  0.0758, -0.5159,  0.3569, -0.1408, -0.1395, -0.2387,  0.7773,
            0.5266, -0.4465]], grad_fn=<AddmmBackward>)
    """
    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, graph, feat, edge_weight=None):
        r"""
        Description
        -----------
        Compute Graph Isomorphism Network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        with graph.local_scope():
            aggregate_fn = self.concat_edge_msg
            # aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))


            diff = torch.tensor(graph.dstdata['neigh'].shape[1: ]) - torch.tensor(feat_dst.shape[1: ])
            zeros = torch.zeros(feat_dst.shape[0], *diff).to(feat_dst.device)
            feat_dst = torch.cat([feat_dst, zeros], dim=1)
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst

    def concat_edge_msg(self, edges):
        if self.edge_feat_loc not in edges.data:
            return {'m': edges.src['h']}
        else:
            m = torch.cat([edges.src['h'], edges.data[self.edge_feat_loc]], dim=1)
            return {'m': m}


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)

        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 graph_pooling_type, neighbor_pooling_type, edge_feat_dim=0,
                 final_dropout=0.0, learn_eps=False, output_dim=1, **kwargs):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """

        super().__init__()
        def init_weights_orthogonal(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
            elif isinstance(m, MLP):
                if hasattr(m, 'linears'):
                    m.linears.apply(init_weights_orthogonal)
                else:
                    m.linear.apply(init_weights_orthogonal)
            elif isinstance(m, nn.ModuleList):
                pass
            else:
                raise Exception()

        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # self.preprocess_nodes = PreprocessNodeAttrs(
        #     node_attrs=node_preprocess, output_dim=node_preprocess_output_dim)
        # print(input_dim)
        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim + edge_feat_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim + edge_feat_dim, hidden_dim, hidden_dim)
            if kwargs['init'] == 'orthogonal':
                init_weights_orthogonal(mlp)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))


        if kwargs['init'] == 'orthogonal':
            self.linears_prediction.apply(init_weights_orthogonal)

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        # h = self.preprocess_nodes(h)
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        return score_over_layer

    def get_graph_embed(self, g, h):
        self.eval()
        with torch.no_grad():
            # return self.forward(g, h).detach().numpy()
            hidden_rep = []
            # h = self.preprocess_nodes(h)
            for i in range(self.num_layers - 1):
                h = self.ginlayers[i](g, h)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                hidden_rep.append(h)

            # perform pooling over all nodes in each graph in every layer
            graph_embed = torch.Tensor([]).to(self.device)
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(g, h)
                graph_embed = torch.cat([graph_embed, pooled_h], dim = 1)

            return graph_embed

    def get_graph_embed_no_cat(self, g, h):
        self.eval()
        with torch.no_grad():
            hidden_rep = []
            # h = self.preprocess_nodes(h)
            for i in range(self.num_layers - 1):
                h = self.ginlayers[i](g, h)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                hidden_rep.append(h)

            # perform pooling over all nodes in each graph in every layer
            # graph_embed = torch.Tensor([]).to(self.device)
            # for i, h in enumerate(hidden_rep):
            #     pooled_h = self.pool(g, h)
            #     graph_embed = torch.cat([graph_embed, pooled_h], dim = 1)

            # return graph_embed
            return self.pool(g, hidden_rep[-1]).to(self.device)

    @property
    def edge_feat_loc(self):
        return self.ginlayers[0].edge_feat_loc

    @edge_feat_loc.setter
    def edge_feat_loc(self, loc):
        for layer in self.ginlayers:
            layer.edge_feat_loc = loc
