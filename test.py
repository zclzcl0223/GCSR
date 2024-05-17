import argparse
import random
import torch
import numpy as np
from utils import *
from model import *
from data import *

def main(args, verbose=True):
    print('Hyper-parameters: \n', args.__dict__)

    model_types = ['MLP', 'APPNP', 'Cheby', 'GCN', 'GraphSage', 'SGC2']
    device = torch.device('cuda:%d' % args.gpu_id)
    data = DataLoader(args.dataset)
    dropout = 0.5 if args.dataset in ['reddit'] else 0
    final_res = {}
    for model_type in model_types:
        
        final_res[model_type] = []

    for exp in range(args.exps):
        syn_feat, syn_label, syn_adj = get_syn_data(args.load_folder, args.dataset, args.reduction_rate, exp)
        syn_label, _ = generate_syn_label(data, args.reduction_rate)
        syn_label = torch.tensor(syn_label)
        syn_feat, syn_label, syn_adj = syn_feat.to(device), syn_label.to(device), syn_adj.to(device)

        if args.dataset in ['ogbn-arxiv']:
            syn_adj[syn_adj < 0.01] = 0

        for model_type in model_types:
            
            print("===feat_%s_%s_%d, testing %s" % (args.dataset, str(args.reduction_rate), exp, model_type))
            net_type = eval(model_type)
            
            if args.dataset in ['reddit'] and model_type in ['SGC2', 'APPNP']:
                with_relu = False
            elif args.dataset in ['ogbn-arxiv'] and args.reduction_rate == 5e-3 and model_type in ['GCN', 'GraphSage', 'SGC2', 'APPNP']:
                with_relu = False
            else:
                with_relu = True
                
            for _ in range(args.runs_test):
                    
                if model_type == 'GAT':
                    # for GAT, we need a sparser adj
                    if args.dataset in ['citeseer']:
                        syn_adj[syn_adj < 0.8] = 0
                    else:
                        syn_adj[syn_adj < 0.5] = 0
                    dropout = 0.1 if args.dataset == 'reddit' else 0
                    weight_decay = 5e-4 if args.dataset in ['reddit', 'citeseer'] else 0
                    net = GAT(nfeat=syn_feat.shape[1], nhid=16, nclass=data.num_classes, 
                              heads=8, dropout=dropout, dataset=args.dataset).to(device)
                    train_iters= 10000 if args.dataset in ['reddit', 'flickr'] else 3000
                    final_res[model_type].append(train_gat_with_eval(args.dataset, net, syn_feat, syn_adj, syn_label, 
                                                                     data, device, lr=1e-3, weight_decay=weight_decay, 
                                                                     train_iters=train_iters, multi_label=False, 
                                                                     verbose=True, with_val=args.with_val))
                else:
                    net = net_type(nfeat=syn_feat.shape[1], nhid=256, nclass=data.num_classes, hop=2, nlayers=2,
                                   dropout=dropout, with_relu=with_relu, with_bn=False).to(device)
                    final_res[model_type].append(train_with_eval(args.dataset, model_type, net, syn_feat, syn_adj, 
                                                                 syn_label, data, device, args.lr_model, args.weight_decay, 
                                                                 train_iters=600, multi_label=False, normalize=args.normalize, 
                                                                 verbose=True, with_val=args.with_val))
    for model_type in model_types:
        final_res[model_type] = np.array(final_res[model_type])
        final_res[model_type] = [final_res[model_type].mean(0), final_res[model_type].std(0)]
    if verbose:
        print('Final result:', final_res)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--load_folder', type=str, default='saved_ours')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--reduction_rate', type=float, default=0.5, help="reduction rate of train set")
    parser.add_argument('--exps', type=int, default=1, help="index of synthetic graph")
    parser.add_argument('--lr_model', type=float, default=1e-2, help="learning rate for training gnn")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay for training gnn")
    parser.add_argument('--runs_test', type=int, default=10, help="repeated test")
    parser.add_argument('--normalize', action="store_true")
    parser.add_argument('--with_val', action="store_true", help='cross validation while training')

    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
