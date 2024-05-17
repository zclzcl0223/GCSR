import os
import random
import torch
import argparse
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import deeprobust.graph.utils as utils
from model import *
from utils import *
from data import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def main(args):
    """
    Main function to train teacher models and save trajectories.
    """
    # Set device
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    nhid = 256
    print('Hyper-parameters: \n', args.__dict__)

    # Standardize dataset name
    args.dataset = args.dataset.lower()
    
    # Create save directory
    save_dir = os.path.join(args.buffer_path, args.dataset, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Load data
    data = DataLoader(args.dataset)
    nclass = data.num_classes
    nfeat = data.num_features
    train_idx = data.train_idx
    test_idx = data.test_idx

    # Data conversion based on dataset and model
    if args.dataset in ['flickr', 'reddit']:
        if args.model_name == 'Cheby':
            _, cheby_adj, _ = data_convert(data, data.train_adj, args.device, idx=train_idx,
                                           normalize=True, is_cheby=True)
        train_feat, train_adj, train_label = data_convert(data, data.train_adj, args.device, 
                                                          idx=train_idx, normalize=True)
        test_feat, test_adj, test_label = data_convert(data, data.test_adj, args.device, 
                                                          idx=test_idx, normalize=True)
    else:
        if args.model_name == 'Cheby':
            _, cheby_adj, _ = data_convert(data, data.full_adj, args.device, normalize=True, 
                                           is_cheby=True)
        train_feat, train_adj, train_label = data_convert(data, data.full_adj, args.device,
                                                          normalize=True)

    print("size of train_adj (%d, %d)" % (train_idx.shape[0], train_idx.shape[0]))
    print("edges in train_adj: %d" % (data.train_adj.sum()))
    
    trajectories = []
    loss = nn.NLLLoss().to(args.device)
    
    for it in range(0, args.num_experts):
        # Initialize teacher model
        teacher_net_type = eval(args.model_name)
        teacher_net = teacher_net_type(nfeat=nfeat, nhid=nhid, nclass=nclass, hop=2, nlayers=args.nlayers,
                                       dropout=0, with_bn=False).to(args.device)
        teacher_net.initialize()
        teacher_net.train()
        
        # Initialize optimizer
        lr = args.lr_teacher
        teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr, weight_decay=args.weight_decay)

        timestamps = []
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        if args.model_name == 'GraphSage':
            if args.dataset in ['flickr', 'reddit']:
                train_loader = get_train_loader(train_adj, torch.arange(train_feat.shape[0]).long())
            else:
                train_loader = get_train_loader(train_adj, train_idx)
        
        for epoch in range(args.train_epochs):
            if epoch == args.train_epochs // 2:
                lr = lr*0.1
                teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr, weight_decay=args.weight_decay)

            teacher_net.train()
            teacher_optim.zero_grad()

            if args.model_name == 'GraphSage':
                for batch_size, n_id, adjs in train_loader:
                    adjs = [adj.to(args.device) for adj in adjs]
                    teacher_optim.zero_grad()
                    out = teacher_net.forward_sampler(train_feat[n_id], adjs)
                    l = loss(out, train_label[n_id[:batch_size]])
                    l.backward()
                    teacher_optim.step()
            else:
                if args.model_name == 'Cheby':
                    y_hat = teacher_net(train_feat, cheby_adj)
                else:
                    y_hat = teacher_net(train_feat, train_adj)
                    
                if args.dataset in ['flickr', 'reddit']:
                    l = loss(y_hat, train_label)
                else:
                    l = loss(y_hat[train_idx], train_label[train_idx])
                
                l.backward()
                teacher_optim.step()
                
            teacher_net.eval()
            if args.dataset in ['flickr', 'reddit']:
                y_hat = teacher_net(test_feat, test_adj)
                loss_test = nn.functional.nll_loss(y_hat, test_label)
                acc_test = utils.accuracy(y_hat, test_label)
            else:
                y_hat = teacher_net(train_feat, train_adj)
                loss_test = nn.functional.nll_loss(y_hat[test_idx], train_label[test_idx])
                acc_test = utils.accuracy(y_hat[test_idx], train_label[test_idx])
            
            print("Epoch %d %s Loss: %.3f  Test acc: %.3f" % (epoch, args.dataset, loss_test, acc_test))
            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])        

        trajectories.append(timestamps)
        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--model', type=str, default='sgc2-lr3-wt54', help='model')
    parser.add_argument('--model_name', type=str, default='SGC2', help='teacher model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=1e-3, help='learning rate for updating network parameters')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--nlayers', type=int, default=2, help='the number of gnn layers')

    args = parser.parse_args()
    main(args)
