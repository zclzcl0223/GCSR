import os
import torch
import copy
import scipy
import numpy as np
import scipy.sparse as sp
import deeprobust.graph.utils as utils
from model import GCN
from torch import nn
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor

def train_with_eval(dataset, net_type, net, syn_feat, syn_adj, syn_label, data, device,
                    lr, weight_decay, train_iters=600, multi_label=False,
                    normalize=True, verbose=True, with_val=False):
    net.initialize()
    features = syn_feat.to(device)
    adj = syn_adj.to(device)
    
    if net_type == 'Cheby':
        adj = adj - torch.eye(adj.shape[0]).to(device)

    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)
    adj = adj_norm
    edge_index = adj.nonzero().T
    edge_weight = adj[edge_index[0], edge_index[1]]

    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       value=edge_weight, sparse_sizes=adj.shape).t()
    labels = syn_label.to(device)

    full_feat, full_adj, _ = data_convert(data, data.full_adj, device, normalize=normalize)
    train_feat, train_adj, train_label = data_convert(data, data.train_adj, device, idx=data.train_idx, normalize=normalize)
    val_feat, val_adj, val_label = data_convert(data, data.val_adj, device, idx=data.val_idx, normalize=normalize)
    test_feat, test_adj, test_label = data_convert(data, data.test_adj, device, idx=data.test_idx, normalize=normalize)

    res = []

    if multi_label:
        loss = torch.nn.BCELoss()
    else:
        loss = torch.nn.NLLLoss()

    labels = labels.float() if multi_label else labels

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc_val = 0
    if net_type == 'GraphSage':
        train_loader = get_train_loader(adj, torch.arange(features.shape[0]).long())
    # Training
    for i in range(train_iters):
        if i == train_iters // 2:
           lr = lr*0.1
           optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        net.train()
        optimizer.zero_grad()
        
        if net_type == 'GraphSage':
            for batch_size, n_id, adjs in train_loader:
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()
                out = net.forward_sampler(features[n_id], adjs)
                loss_train = nn.functional.nll_loss(out, labels[n_id[:batch_size]])
                loss_train.backward()
                optimizer.step()
        else:
            output = net(features, adj)
            loss_train = loss(output, labels)
            loss_train.backward()
            optimizer.step()
        if with_val:
            with torch.no_grad():
                net.eval()
                if dataset in ['flickr', 'reddit']:
                    output = net(val_feat, val_adj)
                    acc_val = utils.accuracy(output, val_label)
                elif dataset in ['cora', 'citeseer', 'ogbn-arxiv']:
                    output = net(full_feat, full_adj)
                    acc_val = utils.accuracy(output[data.val_idx], val_label)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    weights = copy.deepcopy(net.state_dict())
    if with_val:
        net.load_state_dict(weights)
        
    # Eval
    net.eval()
    if dataset in ['flickr', 'reddit']:
        train_output = net(train_feat, train_adj)
        loss_train = nn.functional.nll_loss(train_output, train_label)
        acc_train = utils.accuracy(train_output, train_label)
        res.append(acc_train.item())
        if verbose:
            print("Train set results:",
                    "loss= {:.4f}".format(loss_train.item()),
                    "accuracy= {:.4f}".format(acc_train.item()))
        test_output = net(test_feat, test_adj)
        loss_test = nn.functional.nll_loss(test_output, test_label)
        acc_test = utils.accuracy(test_output, test_label)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()))
    elif dataset in ['cora', 'citeseer', 'ogbn-arxiv']:
        output = net(full_feat, full_adj)
        loss_train = nn.functional.nll_loss(output[data.train_idx], train_label)
        acc_train = utils.accuracy(output[data.train_idx], train_label)
        res.append(acc_train.item())
        if verbose:
            print("Train set results:",
                    "loss= {:.4f}".format(loss_train.item()),
                    "accuracy= {:.4f}".format(acc_train.item()))
        loss_test = nn.functional.nll_loss(output[data.test_idx], test_label)
        acc_test = utils.accuracy(output[data.test_idx], test_label)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()))
    else:
        raise NotImplementedError
    
    return res

def data_convert(data, adj, device, idx=None, normalize=True, is_cheby=False):
    """
    Normalize and move to gpu
    """
    if idx is None:
        feat = data.x
        label = data.y
    else:
        feat = data.x[idx]
        label = data.y[idx]
        
    if is_cheby:
        I = np.eye(feat.shape[0])
        I = sp.csr_matrix(I)
        adj = adj - I

    feat, adj, label = utils.to_tensor(feat, adj, label)
    feat = feat.to(device)
    adj = adj.to(device)
    label = label.to(device)

    # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    if normalize:
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
        adj = adj_norm
    adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                       value=adj._values(), sparse_sizes=adj.size()).t()
    return feat, adj, label

def train_gat_with_eval(dataset, net, syn_feat, syn_adj, syn_label, data, device,
                        lr, weight_decay, train_iters=600, multi_label=False,
                        verbose=True, with_val=False):
    net.initialize()
    features, edge_index, edge_weight, labels = dpr2pyg(syn_feat, syn_adj, syn_label, device)

    full_feat, full_edge_index, full_edge_weight, _ = dpr2pyg(data.x, data.full_adj, data.y, device)
    train_feat, train_edge_index, train_edge_weight, train_label = dpr2pyg(data.x[data.train_idx], data.train_adj, 
                                                                           data.y[data.train_idx], device)
    val_feat, val_edge_index, val_edge_weight, val_label = dpr2pyg(data.x[data.val_idx], data.val_adj, 
                                                                   data.y[data.val_idx], device)
    test_feat, test_edge_index, test_edge_weight, test_label = dpr2pyg(data.x[data.test_idx], data.test_adj, 
                                                                       data.y[data.test_idx], device)

    res = []

    if multi_label:
        loss = torch.nn.BCELoss()
    else:
        loss = torch.nn.NLLLoss()

    labels = labels.float() if multi_label else labels

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc_val = 0
    best_loss_val = 100

    # Training
    for i in range(train_iters):
        if i in [1500]:
           lr = lr*0.1
           optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        net.train()
        optimizer.zero_grad()
        
        output = net(features, edge_index, edge_weight)
        loss_train = loss(output, labels)
        loss_train.backward()
        optimizer.step()
        
        if with_val:
            with torch.no_grad():
                net.eval()
                if dataset in ['flickr', 'reddit']:
                    output = net(val_feat, val_edge_index, val_edge_weight)
                    loss_val = nn.functional.nll_loss(output, val_label)
                    acc_val = utils.accuracy(output, val_label)
                elif dataset in ['cora', 'citeseer', 'ogbn-arxiv']:
                    output = net(full_feat, full_edge_index, full_edge_weight)
                    loss_val = nn.functional.nll_loss(output[data.val_idx], val_label)
                    acc_val = utils.accuracy(output[data.val_idx], val_label)

                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    weights = copy.deepcopy(net.state_dict())
                    
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    weights = copy.deepcopy(net.state_dict())
    if with_val:
        net.load_state_dict(weights)
        
    # Eval
    net.eval()
    if dataset in ['flickr', 'reddit']:
        train_output = net(train_feat, train_edge_index, train_edge_weight)
        loss_train = nn.functional.nll_loss(train_output, train_label)
        acc_train = utils.accuracy(train_output, train_label)
        res.append(acc_train.item())
        if verbose:
            print("Train set results:",
                    "loss= {:.4f}".format(loss_train.item()),
                    "accuracy= {:.4f}".format(acc_train.item()))
        test_output = net(test_feat, test_edge_index, test_edge_weight)
        loss_test = nn.functional.nll_loss(test_output, test_label)
        acc_test = utils.accuracy(test_output, test_label)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()))
    elif dataset in ['cora', 'citeseer', 'ogbn-arxiv']:
        output = net(full_feat, full_edge_index, full_edge_weight)
        loss_train = nn.functional.nll_loss(output[data.train_idx], train_label)
        acc_train = utils.accuracy(output[data.train_idx], train_label)
        res.append(acc_train.item())
        if verbose:
            print("Train set results:",
                    "loss= {:.4f}".format(loss_train.item()),
                    "accuracy= {:.4f}".format(acc_train.item()))
        loss_test = nn.functional.nll_loss(output[data.test_idx], test_label)
        acc_test = utils.accuracy(output[data.test_idx], test_label)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                    "loss= {:.4f}".format(loss_test.item()),
                    "accuracy= {:.4f}".format(acc_test.item()))
    else:
        raise NotImplementedError
    
    return res

def dpr2pyg(feat, adj, label, device):
    if type(adj) == torch.Tensor:
        adj_selfloop = adj.to(device) + torch.eye(adj.shape[0], device=device)
        edge_index_selfloop = adj_selfloop.nonzero().T
        edge_index = edge_index_selfloop.to(device)
        edge_weight = adj_selfloop[edge_index_selfloop[0], edge_index_selfloop[1]]
    else:
        adj_selfloop = adj + sp.eye(adj.shape[0])
        edge_index = torch.LongTensor(np.array(adj_selfloop.nonzero())).to(device)
        edge_weight = torch.FloatTensor(adj_selfloop[adj_selfloop.nonzero()]).to(device)
    
    feat = feat.to(device)
    label = label.to(device)
    
    return feat, edge_index, edge_weight, label

def generate_syn_label(data, reduction_rate):
    """
    Generate synthetic labels whose class distribution are the same as the original labels
    """
    from collections import Counter
    train_label = data.y[data.train_idx].cpu().numpy()
    counter = Counter(train_label)
    num_class_dict = {}
    n = len(train_label)

    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    sum_ = 0
    syn_label = []
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * reduction_rate) - sum_
            syn_label += [c] * num_class_dict[c]
        else:
            num_class_dict[c] = max(int(num * reduction_rate), 1)
            sum_ += num_class_dict[c]
            syn_label += [c] * num_class_dict[c]

    return syn_label, num_class_dict

def get_feat(origin_feat, syn_label, origin_label, train_idx):
    """
    Sampling from the original nodes
    """
    idx_selected = []
    origin_feat = origin_feat.cpu()
    
    from collections import Counter;
    counter = Counter(syn_label.cpu().numpy())  # (class, num)

    for c, n in counter.items():
        tmp = get_idx_selected(c, origin_label[train_idx], n)
        tmp = list(tmp)
        idx_selected = idx_selected + tmp
    idx_selected = np.array(idx_selected).reshape(-1)
    origin_feat = origin_feat[train_idx][idx_selected]

    return origin_feat

def get_idx_selected(c, train_label, num=0):
    """
    Index of the original nodes
    """
    train_label = train_label.cpu()
    class_mask = (train_label == c)
    idx = np.arange(len(train_label))
    idx = idx[class_mask]
    return np.random.permutation(idx)[:num]

def get_syn_adj(syn_feat, P, alpha, beta, E, Q):
    """
    Self-expressive graph structure reconstruction
    """
    XXT = torch.mm(syn_feat, syn_feat.T)
    syn_adj = torch.mm(torch.inverse(XXT + alpha * E + beta * E), (XXT + alpha * P + beta * Q))
    return syn_adj

def get_syn_data(load_folder, dataset, reduction_rate, exp):
    """
    Load synthetic graph
    """
    syn_feat = torch.load(f'{load_folder}/feat_{dataset}_{reduction_rate}_{exp}.pt', map_location='cpu')
    syn_label = torch.load(f'{load_folder}/label_{dataset}_{reduction_rate}_{exp}.pt', map_location='cpu')
    syn_adj = torch.load(f'{load_folder}/adj_{dataset}_{reduction_rate}_{exp}.pt', map_location='cpu')
    return syn_feat, syn_label, syn_adj

def get_train_loader(adj, node_idx):
    """
    Loader for graphsage
    """
    if adj.density() == 1: # it seems that for the synthetic graph, we need the message from all neighbor
        sizes = [len(node_idx), len(node_idx)]
    elif adj.density() > 0.5: # if the weighted graph is too dense, we need a larger neighborhood size
        sizes = [30, 20]
    else:
        sizes = [5, 5]
        
    train_loader = NeighborSampler(adj, node_idx=node_idx,
                                    sizes=sizes, batch_size=len(node_idx),
                                    num_workers=0, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True)

    return train_loader
