import random
import torch
import os
import argparse
import torch_sparse
import numpy as np
import deeprobust.graph.utils as utils
from torch_geometric import transforms as T
from torch import nn
from utils import *
from model import *
from data import *

def main(args, data, exp=None):
    device = torch.device("cuda:%s" % args.gpu_id)
    max_files = 20
    
    test_net_type = eval(args.test_net_type)
    
    # Dataset Information
    nclass = data.num_classes
    nfeat = data.num_features
    nnodes = data.num_nodes
    
    # Dim of synthetic features
    nsyn_feat = nfeat
    
    # Data preprocessing
    train_idx = data.train_idx
    if args.dataset in ['cora', 'citeseer', 'ogbn-arxiv']:
        origin_feat, full_adj, origin_label = data_convert(data, data.full_adj, device, normalize=True)
    elif args.dataset in ['flickr', 'reddit'] :
        origin_feat, full_adj, origin_label = data_convert(data, data.train_adj, device, idx=train_idx, 
                                                           normalize=True)
        train_idx = torch.arange(origin_label.shape[0], dtype=torch.long)
    
    syn_feat_num = int(len(data.train_idx) * args.reduction_rate)
    
    syn_feat = nn.Parameter(torch.FloatTensor(syn_feat_num, nsyn_feat).to(device))
    syn_label, num_class_dict = generate_syn_label(data, args.reduction_rate)
    syn_label = torch.LongTensor(syn_label).flatten().to(device)
    
    # Message passing initialization
    origin_feat_aggregate = origin_feat
    for _ in range(args.message_passing):
        origin_feat_aggregate = torch_sparse.matmul(full_adj, origin_feat_aggregate)
    feat_sub = get_feat(origin_feat_aggregate, syn_label, origin_label, train_idx)
    syn_feat.data.copy_(feat_sub)
    syn_adj = torch.eye(syn_feat_num, dtype=torch.float32, device=device)

    if utils.is_sparse_tensor(syn_adj):
        syn_adj_norm = utils.normalize_adj_tensor(syn_adj, sparse=True)
    else:
        syn_adj_norm = utils.normalize_adj_tensor(syn_adj)
        
    print("syn_feat shape: ", syn_feat.shape)
    print("syn_adj shape: ", syn_adj.shape)
    print("syn_label shape: ", syn_label.shape)
    
    # Training hyperparameters
    optimizer_syn_feat = torch.optim.Adam([syn_feat,], lr=args.lr_feat)
    lr_model = 1e-2
    syn_loss = nn.NLLLoss().to(device)

    # Load expert network parameters
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    expert_dir = os.path.join(expert_dir, args.expert_net)
    print("Expert Dir: {}".format(expert_dir))

    expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))) and n < max_files:
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    file_idx = 0
    expert_idx = 0
    random.shuffle(expert_files)
    # print("loading file {}".format(expert_files[file_idx]))
    buffer = torch.load(expert_files[file_idx])
    random.shuffle(buffer)
        
    print("Condensation begins: Model %s" % args.expert_net_type)
    # Training
    for epoch in range(args.epochs + 1):
        if epoch <= args.epochs:
            student_net_type = eval(args.expert_net_type)
            student_net = student_net_type(nfeat=nfeat, nhid=256, nclass=nclass, hop=args.hop, nlayers=2,
                                        dropout=0, with_bn=False)
            student_net = student_net.to(device)
            student_net.initialize()
            student_net = ReparamModule(student_net)
            student_net.train()
            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
            
            expert_trajectory = buffer[expert_idx]  # Select an expert trajectory
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                # print("loading file {}".format(expert_files[file_idx]))
                del buffer
                buffer = torch.load(expert_files[file_idx])
                random.shuffle(buffer)
            
            start_epoch = np.random.randint(0, args.max_start_epoch)
            starting_params = expert_trajectory[start_epoch]

            target_params = expert_trajectory[start_epoch+args.expert_epochs]
            target_params = torch.cat([p.data.to(device).reshape(-1) for p in target_params], 0)

            # Initialize student network with the starting parameters of the expert trajectory
            student_params = [torch.cat([p.data.to(device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(device).reshape(-1) for p in starting_params], 0)
            
            # Update student network model parameters student_epochs times
            for _ in range(args.student_epochs):
                forward_params = student_params[-1]
                y_syn_hat = student_net(syn_feat, syn_adj_norm, flat_param=forward_params)
                l_syn = syn_loss(y_syn_hat, syn_label)
                grad = torch.autograd.grad(l_syn, student_params[-1], create_graph=True)[0]
                
                student_params.append(student_params[-1] - lr_model * grad)  # Gradient descent

            # Gradient matching
            param_loss = torch.tensor(0.0).to(device)
            param_dist = torch.tensor(0.0).to(device)

            param_loss += nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            param_loss /= num_params
            param_dist /= num_params
            param_loss /= param_dist

            grand_loss = param_loss

            # Update
            optimizer_syn_feat.zero_grad()
            grand_loss.backward()
            optimizer_syn_feat.step()

        for _ in student_params:
            del _
        
        if args.test:
            if epoch % args.eval_interval == 0:
                print('Epoch {}, grand_loss: {}'.format(epoch, grand_loss.item()))
            eval_epochs = np.arange(0, args.epochs+1, args.eval_interval)
            
            if epoch in eval_epochs:
                res = []
                syn_feat_test, syn_label_test = syn_feat.detach(), syn_label
                syn_adj_test = syn_adj.detach()
                
                for i in range(args.runs):
                    net = test_net_type(nfeat=syn_feat_test.shape[1], nhid=args.hidden_test, nclass=nclass, 
                                        nlayers=args.layer_test, dropout=args.dropout_test, with_relu=True, with_bn=False)
                    net = net.to(device)
                    res.append(train_with_eval(args.dataset, args.test_net_type, net, syn_feat_test, syn_adj_test, syn_label_test, 
                                                data, device, args.lr_model, args.weight_decay, train_iters=600, multi_label=False, 
                                                normalize=args.normalize, verbose=False, with_val=args.with_val))
                res = np.array(res)
                print('Train/Test Mean Accuracy:',
                        repr([res.mean(0), res.std(0)]))
                
    res = []
    syn_feat_test, syn_label_test = syn_feat.detach(), syn_label
    syn_adj_test = syn_adj.detach()
    
    for _ in range(args.runs_test):
        net = test_net_type(nfeat=syn_feat_test.shape[1], nhid=args.hidden_test, nclass=nclass, nlayers=args.layer_test,
                            dropout=args.dropout_test, with_relu=True, with_bn=False).to(device)
        res.append(train_with_eval(args.dataset, args.test_net_type, net, syn_feat_test, syn_adj_test, syn_label_test, data, 
                                   device, args.lr_model, args.weight_decay, train_iters=600, multi_label=False, 
                                   normalize=args.normalize, verbose=False, with_val=args.with_val))
    
    if args.save:
        torch.save(syn_feat_test, f'{args.saved_folder}/feat_{args.dataset}_{args.reduction_rate}_{exp}.pt')
        torch.save(syn_adj_test, f'{args.saved_folder}/adj_{args.dataset}_{args.reduction_rate}_{exp}.pt')
        torch.save(syn_label_test, f'{args.saved_folder}/label_{args.dataset}_{args.reduction_rate}_{exp}.pt')
    
    return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='expert path')
    parser.add_argument('--expert_net', type=str, default='sgc2-lr3-wt54', help='name of expert net')
    parser.add_argument('--expert_net_type', type=str, default='SGC2', help='teacher model')
    parser.add_argument('--test_net_type', type=str, default='GCN', help='test model')
    parser.add_argument('--student_epochs', type=int, default=5)
    parser.add_argument('--max_start_epoch', type=int, default=60)
    parser.add_argument('--expert_epochs', type=int, default=2)
    parser.add_argument('--lr_feat', type=float, default=1e-6, help='lr for updating synthetic data')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=8000)
    parser.add_argument('--eval_interval', type=int, default=400)
    parser.add_argument('--hidden_test', type=int, default=256, help='size of hidden layer for test net')
    parser.add_argument('--layer_test', type=int, default=2, help='number of layers for test net')
    parser.add_argument('--lr_model', type=float, default=1e-2, help='learning rate for training test net')
    parser.add_argument('--dropout_test', type=float, default=0.5, help='dropout rate for test net')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--hop', type=int, default=2, help='receptive field')
    parser.add_argument('--hidden', type=int, default=256, help='size of hidden layer for condensation model')
    parser.add_argument('--reduction_rate', type=float, default=0.5, help="reduction rate of train set")
    parser.add_argument('--exps', type=int, default=1, help="repeated experiments")
    parser.add_argument('--runs', type=int, default=5, help="repeated evaluation")
    parser.add_argument('--runs_test', type=int, default=10, help="repeated test")
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--saved_folder', type=str, default='saved_ours')
    parser.add_argument('--normalize', action="store_true", help='normalize adjacency matrix')
    parser.add_argument('--test', action="store_true", help='test while condensing')
    parser.add_argument('--with_val', action="store_true", help='cross validation while training gnns')
    parser.add_argument('--message_passing', type=int, default=2, help='message passing initialization')
    args = parser.parse_args()
    
    print('Hyper-parameters: \n', args.__dict__)
    
    if not os.path.exists(args.saved_folder):
        os.makedirs(args.saved_folder)
        
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    data = DataLoader(args.dataset)
    res = []

    for exp in range(args.exps):
        print("Exp %d Begin:" % exp)
        res += main(args, data, exp=exp)

    res = np.array(res)
    print('\nFinal Train/Test Mean Accuracy:',
            repr([res.mean(0), res.std(0)]))
    