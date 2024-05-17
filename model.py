import torch
import math
import scipy as sp
import numpy as np
import torch_sparse
from typing import Union, Tuple, Optional
from torch import nn
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from contextlib import contextmanager
from deeprobust.graph import utils
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

#MLP
class MLP(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, hop=None, nlayers=2, dropout=0.5, with_relu=True, 
                 with_bias=True, with_bn=False):
        super().__init__()
        self.weight1 = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias1 = nn.Parameter(torch.FloatTensor(nhid)) if with_bias else None
        self.weight2 = nn.Parameter(torch.FloatTensor(nhid, nclass))
        self.bias2 = nn.Parameter(torch.FloatTensor(nclass)) if with_bias else None
        self.bns = nn.BatchNorm1d(nhid) if with_bn else None
        self.with_bias = with_bias
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.dropout = dropout
        self.initialize()

    def initialize(self):
        stdv1 = 1. / math.sqrt(self.weight1.T.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.T.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.with_bias:
            self.bias1.data.uniform_(-stdv1, stdv1)
            self.bias2.data.uniform_(-stdv2, stdv2)
        if self.with_bn:
            self.bns.reset_parameters()

    def encoder(self, x):
        x = torch.mm(x, self.weight1)
        x += self.bias1 if self.with_bias else 0
        x = self.bns(x) if self.with_bn else x
        x = torch.relu(x) if self.with_relu else x
        x = nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.weight2)
        x += self.bias2 if self.with_bias else 0
    
        return x
    
    def forward(self, x, adj):
        x = torch.mm(x, self.weight1)
        x += self.bias1 if self.with_bias else 0
        x = self.bns(x) if self.with_bn else x
        x = torch.relu(x) if self.with_relu else x
        x = nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.weight2)
        x += self.bias2 if self.with_bias else 0
    
        return nn.functional.log_softmax(x, dim=1) 

    def forward_sampler(self, x, adjs):
        x = torch.mm(x, self.weight1)
        x += self.bias1 if self.with_bias else 0
        x = self.bns(x) if self.with_bn else x
        x = torch.relu(x) if self.with_relu else x
        x = nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.weight2)
        x += self.bias2 if self.with_bias else 0
        
        return nn.functional.log_softmax(x, dim=1)

#SGC1
class SGC1(nn.Module):

    def __init__(self, nfeat, nclass, hop=2, nlayers=1, nhid=None, dropout=None, with_relu=None, 
                 with_bias=True, with_bn=None):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(nfeat, nclass))
        self.bias = nn.Parameter(torch.FloatTensor(nclass)) if with_bias else None
        self.hop = hop
        self.with_bias = with_bias
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.uniform_(-stdv, stdv)
   
    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        for i in range(self.hop):
            if isinstance(adj, torch_sparse.SparseTensor):
                x = torch_sparse.matmul(adj, x)
            else:
                x = torch.mm(adj, x)
        x += self.bias if self.with_bias else 0
        return nn.functional.log_softmax(x, dim=1)
    
    def forward_sampler(self, x, adjs):
        x = torch.mm(x, self.weight)
        for ix, (adj, _, size) in enumerate(adjs):
            if isinstance(adj, torch_sparse.SparseTensor):
                x = torch_sparse.matmul(adj, x)
            else:
                x = torch.mm(adj, x)
        x += self.bias if self.with_bias else 0
        return nn.functional.log_softmax(x, dim=1)

#SGC2  
class SGC2(nn.Module):

    def __init__(self, nfeat, nhid, nclass, hop=2, nlayers=2, dropout=0.5, with_relu=False, 
                 with_bias=True, with_bn=None):
        super().__init__()
        self.weight1 = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        self.bias1 = nn.Parameter(torch.FloatTensor(nhid)) if with_bias else None
        self.weight2 = nn.Parameter(torch.FloatTensor(nhid, nclass))
        self.bias2 = nn.Parameter(torch.FloatTensor(nclass)) if with_bias else None
        self.hop = hop
        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.initialize()

    def initialize(self):
        stdv1 = 1. / math.sqrt(self.weight1.T.size(1))
        stdv2 = 1. / math.sqrt(self.weight2.T.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        self.weight2.data.uniform_(-stdv2, stdv2)
        if self.with_bias:
            self.bias1.data.uniform_(-stdv1, stdv1)
            self.bias2.data.uniform_(-stdv2, stdv2)

    def forward(self, x, adj):
        x = torch.mm(x, self.weight1)
        x += self.bias1 if self.with_bias else 0
        x = torch.relu(x) if self.with_relu else x
        x = nn.functional.dropout(x, self.dropout, training=self.training)

        x = torch.mm(x, self.weight2)
        x += self.bias2 if self.with_bias else 0
        for i in range(self.hop):
            if isinstance(adj, torch_sparse.SparseTensor):
                x = torch_sparse.matmul(adj, x)
            else:
                x = torch.mm(adj, x)
    
        return nn.functional.log_softmax(x, dim=1) 

    def forward_sampler(self, x, adjs):
        x = torch.mm(x, self.weight1)
        x += self.bias1 if self.with_bias else 0
        x = torch.relu(x) if self.with_relu else x
        x = nn.functional.dropout(x, self.dropout, training=self.training)

        x = torch.mm(x, self.weight2)
        x += self.bias2 if self.with_bias else 0
        for ix, (adj, _, size) in enumerate(adjs):
            if isinstance(adj, torch_sparse.SparseTensor):
                x = torch_sparse.matmul(adj, x)
            else:
                x = torch.mm(adj, x)
            
        return nn.functional.log_softmax(x, dim=1)

#GCN
class GraphConvolution(nn.Module):
    
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.mm(input, self.weight)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, output)
        else:
            output = torch.spmm(adj, output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, hop=2, nlayers=2, dropout=0.5, with_relu=True,
                 with_bias=True, with_bn=False):
        super().__init__()
        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))

        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias

    def initialize(self):
        """
        Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)

        return nn.functional.log_softmax(x, dim=1)

    def embed(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)

        return x

    def forward_sampler(self, x, adjs):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)

        return nn.functional.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, (adj) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)

        return nn.functional.log_softmax(x, dim=1)

#GraphSage
class SageConvolution(nn.Module):

    def __init__(self, in_features, out_features, with_bias=True, root_weight=False):
        super(SageConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_l = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias_l = nn.Parameter(torch.FloatTensor(out_features))
        if root_weight:
            self.weight_r = nn.Parameter(torch.FloatTensor(in_features, out_features))
            self.bias_r = nn.Parameter(torch.FloatTensor(out_features))
        self.root_weight = root_weight
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_l.T.size(1))
        self.weight_l.data.uniform_(-stdv, stdv)
        self.bias_l.data.uniform_(-stdv, stdv)
        
        if self.root_weight:
            stdv = 1. / math.sqrt(self.weight_r.T.size(1))
            self.weight_r.data.uniform_(-stdv, stdv)
            self.bias_r.data.uniform_(-stdv, stdv)
        
    def forward(self, input, adj, size=None):
        """ 
        Graph Convolutional Layer forward function
        """
        output = torch.mm(input, self.weight_l)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, output)
        else:
            output = torch.spmm(adj, output)
        output = output + self.bias_l

        if self.root_weight:
            if size is not None:
                output = output + input[:size[1]] @ self.weight_r + self.bias_r
            else:
                output = output + input @ self.weight_r + self.bias_r
        else:
            output = output

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphSage(nn.Module):

    def __init__(self, nfeat, nhid, nclass, hop=None, nlayers=2, dropout=0.5, 
                 with_relu=True, with_bias=True, with_bn=False):
        super(GraphSage, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(SageConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(SageConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(SageConvolution(nhid, nclass, with_bias=with_bias))
        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)
                
        return nn.functional.log_softmax(x, dim=1)
    
    def forward_sampler(self, x, adjs):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj, size=size)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)

        return nn.functional.log_softmax(x, dim=1)

    def initialize(self):
        """
        Initialize parameters of SageConv.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

#ChebyNet
class ChebConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """
    def __init__(self, in_features, out_features, with_bias=True, single_param=True, K=2):
        """set single_param to True to alleivate the overfitting issue"""
        super(ChebConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lins = torch.nn.ModuleList([
           MyLinear(in_features, out_features, with_bias=False) for _ in range(K)])
        # self.lins = torch.nn.ModuleList([
        #    MyLinear(in_features, out_features, with_bias=True) for _ in range(K)])
        if with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.single_param = single_param
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, input, adj, size=None):
        """ 
        Graph Convolutional Layer forward function
        """
        # support = torch.mm(input, self.weight_l)
        x = input
        Tx_0 = x[:size[1]] if size is not None else x
        Tx_1 = x # dummy
        output = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            if isinstance(adj, torch_sparse.SparseTensor):
                Tx_1 = torch_sparse.matmul(adj, x)
            else:
                Tx_1 = torch.spmm(adj, x)

            if self.single_param:
                output = output + self.lins[0](Tx_1)
            else:
                output = output + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            if self.single_param:
                lin = self.lins[0]
            if isinstance(adj, torch_sparse.SparseTensor):
                Tx_2 = torch_sparse.matmul(adj, Tx_1)
            else:
                Tx_2 = torch.spmm(adj, Tx_1)
            Tx_2 = 2. * Tx_2 - Tx_0
            output = output + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Cheby(nn.Module):

    def __init__(self, nfeat, nhid, nclass, hop=None, nlayers=2, dropout=0.5,
                 with_relu=True, with_bias=True, with_bn=False):
        super(Cheby, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(ChebConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(ChebConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(ChebConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(ChebConvolution(nhid, nclass, with_bias=with_bias))

        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x  = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)

        return nn.functional.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj, size=size)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)

        return nn.functional.log_softmax(x, dim=1)
        
    def initialize(self):
        """
        Initialize parameters of Cheby.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

#APPNP
class APPNP(nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, hop=None, nlayers=2, dropout=0.5,
                 with_relu=True, with_bias=True, with_bn=False):
        super(APPNP, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass
        self.alpha = 0.1

        if with_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(nhid))

        self.layers = nn.ModuleList([])
        self.layers.append(MyLinear(nfeat, nhid))
        self.layers.append(MyLinear(nhid, nclass))

        self.nlayers = nlayers
        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)
        h = x
        # here nlayers means K
        for i in range(self.nlayers):
            adj_drop = adj
            if isinstance(adj_drop, torch_sparse.SparseTensor):
                x = torch_sparse.matmul(adj_drop, x)
            else:
                x = torch.spmm(adj_drop, x)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h
            
        return nn.functional.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = torch.relu(x)
                x = nn.functional.dropout(x, self.dropout, training=self.training)

        h = x
        for ix, (adj, _, size) in enumerate(adjs):
            adj_drop = adj
            h = h[: size[1]]
            if isinstance(adj_drop, torch_sparse.SparseTensor):
                x = torch_sparse.matmul(adj_drop, x)
            else:
                x = torch.spmm(adj_drop, x)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h
            
        return nn.functional.log_softmax(x, dim=1)

    def initialize(self):
        """
        Initialize parameters of APPNP.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

class MyLinear(nn.Module):
    """
    Simple Linear layer, modified from https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = nn.Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = nn.Linear(in_channels[1], heads * out_channels, False)

        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()
        self.edge_weight = None

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, edge_weight=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

                if edge_weight is not None:
                    if self.edge_weight is None:
                        self.edge_weight = edge_weight

                    if edge_index.size(1) != self.edge_weight.shape[0]:
                        self.edge_weight = edge_weight

            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)

        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = nn.functional.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = nn.functional.dropout(alpha, p=self.dropout, training=self.training)

        if self.edge_weight is not None:
            x_j = self.edge_weight.view(-1, 1, 1) * x_j
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GAT(torch.nn.Module):
    
    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, 
                 with_bias=True, **kwargs):
        super(GAT, self).__init__()
        self.dropout = dropout
        if 'dataset' in kwargs:
            if kwargs['dataset'] in ['ogbn-arxiv']:
                dropout = 0.7 # arxiv
            elif kwargs['dataset'] in ['reddit']:
                dropout = 0.05
            elif kwargs['dataset'] in ['flickr']:
                dropout = 0.8
            else:
                dropout = 0.7 # cora, citeseer
        else:
            dropout = 0.7
        self.conv1 = GATConv(
            nfeat,
            nhid,
            heads=heads,
            dropout=dropout,
            bias=with_bias)

        self.conv2 = GATConv(
            nhid * heads,
            nclass,
            heads=output_heads,
            concat=False,
            dropout=dropout,
            bias=with_bias)

    def forward(self, x, edge_index, edge_weight):
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = nn.functional.elu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        
        return nn.functional.log_softmax(x, dim=1)

    def initialize(self):
        """
        Initialize parameters of GAT.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
                           
class ReparamModule(nn.Module):
    """
    ReparamModule, similar to https://github.com/GeorgeCazenavette/mtt-distillation/blob/main/reparam_module.py
    """
    def _get_module_from_name(self, mn):
        if mn == '':
            return self
        m = self
        for p in mn.split('.'):
            m = getattr(m, p)
        return m

    def __init__(self, module):
        super(ReparamModule, self).__init__()
        self.module = module

        param_infos = []  # (module name/path, param name)
        shared_param_memo = {}
        shared_param_infos = []  # (module name/path, param name, src module name/path, src param_name)
        params = []
        param_numels = []
        param_shapes = []
        for mn, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    if p in shared_param_memo:
                        shared_mn, shared_n = shared_param_memo[p]
                        shared_param_infos.append((mn, n, shared_mn, shared_n))
                    else:
                        shared_param_memo[p] = (mn, n)
                        param_infos.append((mn, n))
                        params.append(p.detach())
                        param_numels.append(p.numel())
                        param_shapes.append(p.size())

        assert len(set(p.dtype for p in params)) <= 1, \
            "expects all parameters in module to have same dtype"

        # store the info for unflatten
        self._param_infos = tuple(param_infos)
        self._shared_param_infos = tuple(shared_param_infos)
        self._param_numels = tuple(param_numels)
        self._param_shapes = tuple(param_shapes)

        # flatten
        flat_param = nn.Parameter(torch.cat([p.reshape(-1) for p in params], 0))
        self.register_parameter('flat_param', flat_param)
        self.param_numel = flat_param.numel()
        del params
        del shared_param_memo

        # deregister the names as parameters
        for mn, n in self._param_infos:
            delattr(self._get_module_from_name(mn), n)
        for mn, n, _, _ in self._shared_param_infos:
            delattr(self._get_module_from_name(mn), n)

        # register the views as plain attributes
        self._unflatten_param(self.flat_param)

        # now buffers
        # they are not reparametrized. just store info as (module, name, buffer)
        buffer_infos = []
        for mn, m in self.named_modules():
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    buffer_infos.append((mn, n, b))

        self._buffer_infos = tuple(buffer_infos)
        self._traced_self = None

    def trace(self, example_input, **trace_kwargs):
        assert self._traced_self is None, 'This ReparamModule is already traced'

        if isinstance(example_input, torch.Tensor):
            example_input = (example_input,)
        example_input = tuple(example_input)
        example_param = (self.flat_param.detach().clone(),)
        example_buffers = (tuple(b.detach().clone() for _, _, b in self._buffer_infos),)

        self._traced_self = torch.jit.trace_module(
            self,
            inputs=dict(
                _forward_with_param=example_param + example_input,
                _forward_with_param_and_buffers=example_param + example_buffers + example_input,
            ),
            **trace_kwargs,
        )

        # replace forwards with traced versions
        self._forward_with_param = self._traced_self._forward_with_param
        self._forward_with_param_and_buffers = self._traced_self._forward_with_param_and_buffers
        return self

    def clear_views(self):
        for mn, n in self._param_infos:
            setattr(self._get_module_from_name(mn), n, None)  # This will set as plain attr

    def _apply(self, *args, **kwargs):
        if self._traced_self is not None:
            self._traced_self._apply(*args, **kwargs)
            return self
        return super(ReparamModule, self)._apply(*args, **kwargs)

    def _unflatten_param(self, flat_param):
        ps = (t.view(s) for (t, s) in zip(flat_param.split(self._param_numels), self._param_shapes))
        for (mn, n), p in zip(self._param_infos, ps):
            setattr(self._get_module_from_name(mn), n, p)  # This will set as plain attr
        for (mn, n, shared_mn, shared_n) in self._shared_param_infos:
            setattr(self._get_module_from_name(mn), n, getattr(self._get_module_from_name(shared_mn), shared_n))

    @contextmanager
    def unflattened_param(self, flat_param):
        saved_views = [getattr(self._get_module_from_name(mn), n) for mn, n in self._param_infos]
        self._unflatten_param(flat_param)
        yield
        # Why not just `self._unflatten_param(self.flat_param)`?
        # 1. because of https://github.com/pytorch/pytorch/issues/17583
        # 2. slightly faster since it does not require reconstruct the split+view
        #    graph
        for (mn, n), p in zip(self._param_infos, saved_views):
            setattr(self._get_module_from_name(mn), n, p)
        for (mn, n, shared_mn, shared_n) in self._shared_param_infos:
            setattr(self._get_module_from_name(mn), n, getattr(self._get_module_from_name(shared_mn), shared_n))

    @contextmanager
    def replaced_buffers(self, buffers):
        for (mn, n, _), new_b in zip(self._buffer_infos, buffers):
            setattr(self._get_module_from_name(mn), n, new_b)
        yield
        for mn, n, old_b in self._buffer_infos:
            setattr(self._get_module_from_name(mn), n, old_b)

    def _forward_with_param_and_buffers(self, flat_param, buffers, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            with self.replaced_buffers(buffers):
                return self.module(*inputs, **kwinputs)

    def _forward_with_param(self, flat_param, *inputs, **kwinputs):
        with self.unflattened_param(flat_param):
            return self.module(*inputs, **kwinputs)

    def forward(self, *inputs, flat_param=None, buffers=None, **kwinputs):
        flat_param = torch.squeeze(flat_param)
        # print("PARAMS ON DEVICE: ", flat_param.get_device())
        # print("DATA ON DEVICE: ", inputs[0].get_device())
        # flat_param.to("cuda:{}".format(inputs[0].get_device()))
        # self.module.to("cuda:{}".format(inputs[0].get_device()))
        if flat_param is None:
            flat_param = self.flat_param
        if buffers is None:
            return self._forward_with_param(flat_param, *inputs, **kwinputs)
        else:
            return self._forward_with_param_and_buffers(flat_param, tuple(buffers), *inputs, **kwinputs)
