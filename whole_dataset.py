import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import deeprobust.graph.utils as utils
from model import *
from utils import *
from data import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main_transductive(args):
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Hyper-parameters: \n', args.__dict__)
    nhidden = 256
    # Load dataset
    data = DataLoader(args.dataset)
    train_idx = data.train_idx 
    test_idx = data.test_idx
    val_idx = data.val_idx

    print("size of train_adj (%d, %d)" % (train_idx.shape[0], train_idx.shape[0]))
    print("edges in train_adj: %d" % (data.train_adj.sum()))
    
    # Data preprocessing
    if args.model_name == 'Cheby':
        _, cheby_adj, _ = data_convert(data, data.full_adj, args.device, normalize=True, 
                                        is_cheby=True)
    full_feat, full_adj, full_label = data_convert(data, data.full_adj, args.device,
                                                   normalize=True)

    full_acc = []
    loss = nn.NLLLoss().to(args.device)

    for it in range(0, args.exps):
        model_type = eval(args.model_name)
        net = model_type(nfeat=data.num_features, nhid=nhidden, nclass=data.num_classes, nlayers=2,
                         dropout=args.dropout, with_bn=False).to(args.device)
        net.initialize()
        lr = args.lr
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)
        best_acc_val = 0

        if args.model_name == 'GraphSage':
            train_loader = get_train_loader(full_adj, train_idx)
        
        for epoch in range(args.train_epochs):
            if epoch == args.train_epochs // 2:
                lr = lr*0.1
                optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)
            
            net.train()
            optimizer.zero_grad()
            
            if args.model_name == 'GraphSage':
                for batch_size, n_id, adjs in train_loader:
                    adjs = [adj.to(args.device) for adj in adjs]
                    optimizer.zero_grad()
                    out = net.forward_sampler(full_feat[n_id], adjs)
                    l = loss(out, full_label[n_id[:batch_size]])
                    l.backward()
                    optimizer.step()
            else:
                if args.model_name == 'Cheby':
                    y_hat = net(full_feat, cheby_adj)
                else:
                    y_hat = net(full_feat, full_adj)
                    
                l = loss(y_hat[train_idx], full_label[train_idx])
                l.backward()
                optimizer.step()
                
            if args.with_val:
                with torch.no_grad():
                    net.eval()
                    output = net(full_feat, full_adj)

                    acc_val = utils.accuracy(output[val_idx], full_label[val_idx])

                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        weights = copy.deepcopy(net.state_dict())
                        
        if args.with_val:
            net.load_state_dict(weights)

        # Test
        net.eval()
        y_hat = net(full_feat, full_adj)
        acc_test = utils.accuracy(y_hat[test_idx], full_label[test_idx])
        # print("%s Test acc: %.3f" % (args.dataset, acc_test))
        full_acc.append(acc_test.cpu())
        
    full_acc = np.array(full_acc)
    print("full mean/max acc/std: %.3f %.3f %.3f" % (full_acc.mean(), full_acc.max(), full_acc.std()))

def main_inductive(args):
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Hyper-parameters: \n', args.__dict__)
    nhid = 256
    
    # Load dataset
    data = DataLoader(args.dataset)
    train_idx = data.train_idx
    test_idx = data.test_idx
    val_idx = data.val_idx
    print("size of train_adj (%d, %d)" % (train_idx.shape[0], train_idx.shape[0]))
    print("edges in train_adj: %d" % (data.train_adj.sum()))
    
    # Data preprocessing
    if args.model_name == 'Cheby':
        _, cheby_adj, _ = data_convert(data, data.train_adj, args.device, idx=train_idx,
                                       normalize=True, is_cheby=True)
    train_feat, train_adj, train_label = data_convert(data, data.train_adj, args.device, 
                                                          idx=train_idx, normalize=True)
    test_feat, test_adj, test_label = data_convert(data, data.test_adj, args.device, 
                                                          idx=test_idx, normalize=True)
    val_feat, val_adj, val_label = data_convert(data, data.val_adj, args.device, 
                                                          idx=val_idx, normalize=True)
    full_acc = []
    loss = nn.NLLLoss().to(args.device)

    for it in range(0, args.exps):
        model_type =eval(args.model_name)
        net = model_type(nfeat=data.num_features, nhid=nhid, nclass=data.num_classes, nlayers=2,
                         dropout=args.dropout, with_relu=True, with_bn=False).to(args.device)
        net.initialize()
        lr = args.lr
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)
        best_acc_val = 0
        
        if args.model_name == 'GraphSage':
            train_loader = get_train_loader(train_adj, torch.arange(train_feat.shape[0]).long())
        
        for epoch in range(args.train_epochs):

            if epoch == args.train_epochs // 2:
                lr = lr*0.1
                optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)
            net.train()
            optimizer.zero_grad()
            
            if args.model_name == 'GraphSage':
                for batch_size, n_id, adjs in train_loader:
                    adjs = [adj.to(args.device) for adj in adjs]
                    optimizer.zero_grad()
                    out = net.forward_sampler(train_feat[n_id], adjs)
                    l = loss(out, train_label[n_id[:batch_size]])
                    l.backward()
                    optimizer.step()
            else:
                if args.model_name == 'Cheby':
                    y_hat = net(train_feat, cheby_adj)
                else:
                    y_hat = net(train_feat, train_adj)
                l = loss(y_hat, train_label)
                l.backward()
                optimizer.step()
                
            if args.with_val:
                with torch.no_grad():
                    net.eval()
                    output = net(val_feat, val_adj)

                    acc_val = utils.accuracy(output, val_label)

                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        weights = copy.deepcopy(net.state_dict())
                        
            net.eval()
            y_hat = net(test_feat, test_adj)
            loss_test = nn.functional.nll_loss(y_hat, test_label)
            acc_test = utils.accuracy(y_hat, test_label)
            # print("Epoch %d %s Loss: %.3f  Test acc: %.3f" % (epoch, args.dataset, loss_test, acc_test))
       
        if args.with_val:
            net.load_state_dict(weights)
            
        # Test
        net.eval()
        y_hat = net(test_feat, test_adj)
        acc_test = utils.accuracy(y_hat, test_label)
        # print("%s Test acc: %.3f" % (args.dataset, acc_test))
        full_acc.append(acc_test.cpu())
        
    full_acc = np.array(full_acc)
    print("whole mean/max acc/std: %.3f %.3f %.3f" % (full_acc.mean(), full_acc.max(), full_acc.std()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--exps', type=int, default=10, help='repeated experiments')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for updating network parameters')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--model_name', type=str, default='GCN', help='model')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_epochs', type=int, default=600)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='l2 regularization')
    parser.add_argument('--with_val', action="store_true", help='cross validation while training')

    args = parser.parse_args()
    args.dataset = args.dataset.lower()
    if args.dataset in ['cora', 'citeseer', 'ogbn-arxiv']:
        main_transductive(args)
    elif args.dataset in ['flickr', 'reddit']:
        main_inductive(args)
    else:
        raise NotImplementedError
