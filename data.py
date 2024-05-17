import torch
import json
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import StandardScaler
from torch_geometric import transforms as T

class DataLoader():
    """
    Data loading and preprocessing   
    """
    def __init__(self, name):
        """
        Initialize DataLoader with the given dataset name.

        Args:
            name (str): Name of the dataset.
        """
        self.name = name
        name = name.lower()
        path = './' + 'data/' + name
        if name in ['cora', 'citeseer', 'pubmed']:
            data = Planetoid(root=path, name=name, transform=T.NormalizeFeatures())[0]
            self.num_nodes = data.num_nodes
            self.full_adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]),
                                         (data.edge_index[0], data.edge_index[1])), 
                                          shape=(self.num_nodes, self.num_nodes))
            self.train_idx = torch.nonzero(data.train_mask).flatten()
            self.val_idx = torch.nonzero(data.val_mask).flatten()
            self.test_idx = torch.nonzero(data.test_mask).flatten()
            self.x = data.x
            self.y = data.y.reshape(-1)
        elif name in ['ogbn-arxiv', 'reddit', 'flickr']:
            self.full_adj = sp.load_npz(path + '/' + 'adj_full.npz')
            if name == 'ogbn-arxiv':
                self.full_adj.data = np.array(self.full_adj.data, dtype=np.float32)
                self.full_adj = self.full_adj + self.full_adj.T
                self.full_adj[self.full_adj > 1] = 1

            self.num_nodes = self.full_adj.shape[0]
            role = json.load(open(path + '/' + 'role.json', 'r'))
            self.train_idx = torch.tensor(role['tr'], dtype=torch.int64)
            self.val_idx = torch.tensor(role['va'], dtype=torch.int64)
            self.test_idx = torch.tensor(role['te'], dtype=torch.int64)
            self.x = np.load(path + '/' + 'feats.npy')
            train_nodes = self.x[self.train_idx]
            scaler = StandardScaler()
            scaler.fit(train_nodes)
            self.x = torch.tensor(scaler.transform(self.x), dtype=torch.float32)
            class_map = json.load(open(path + '/' + 'class_map.json','r'))
            self.y = torch.tensor(self.process_labels(class_map), dtype=torch.int64)
        else:
            raise NotImplementedError

        self.train_adj = self.full_adj[np.ix_(self.train_idx, self.train_idx)]  # Training set subgraph
        self.val_adj = self.full_adj[np.ix_(self.val_idx, self.val_idx)]  # Validation set subgraph
        self.test_adj = self.full_adj[np.ix_(self.test_idx, self.test_idx)]  # Test set subgraph
        self.num_edges = self.full_adj.sum()  
        self.num_features = self.x.shape[1]
        self.num_classes = self.y.max().item() + 1
        self.edge_probability = None

    def probability_map(self):
        """
        Calculate edge probability based on class distribution.

        Returns:
            None
        """
        if self.name in ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv']:
            adj = self.full_adj.todense()
            y = self.y
        elif self.name in ['reddit', 'flickr']:
            adj = self.train_adj.todense()
            y = self.y[self.train_idx]
                    
        edge_probability = torch.zeros([self.num_classes, self.num_classes], dtype=torch.float32)
        for c in range(self.num_classes):
            node_c = torch.Tensor(adj[y == c])
            for i in range(self.num_classes):
                edge_probability[c][i] = torch.sum(node_c[:, y==i])
        
        self.edge_probability = edge_probability
    
    def print_info(self):
        """
        Print dataset information.

        Returns:
            None
        """
        print("num_nodes: ", self.num_nodes, 
              " num_edges: ", self.num_edges, 
              " num_classes: ", self.num_classes, 
              " num_features: ", self.num_features, 
              "\ntrain_idx: ",  self.train_idx.shape, 
              " val_mask: ", self.val_idx.shape, 
              " test_mask: ", self.test_idx.shape)

    def process_labels(self, class_map):
        """
        Process labels and convert them to an array.

        Args:
            class_map (dict): Mapping of node indices to class labels.

        Returns:
            class_arr (np.array): Processed class labels array.
        """
        num_vertices = self.num_nodes
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.nclass = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k,v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int)
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.nclass = max(class_arr) + 1
        return class_arr
    
def idx_to_mask(idx, size):
    """
    Convert indices to a boolean mask.

    Args:
        idx (torch.Tensor): Tensor containing indices.
        size (int): Size of the mask.

    Returns:
        mask (torch.Tensor): Boolean mask.
    """
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[idx] = 1
    return mask

def to_undirected(edge_index, num_nodes):
    """
    Convert the edge_index to an undirected version (applicable only to ogbn-arxiv).

    Args:
        edge_index (torch.Tensor): Edge indices.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        torch.concat([edge_index, edge_index_reverse], dim=-1) (torch.Tensor): Undirected edge indices.
    """
    edge_index_reverse = torch.zeros(edge_index.shape, dtype=edge_index.dtype)
    edge_index_reverse[0] = edge_index[1]
    edge_index_reverse[1] = edge_index[0]
    return torch.concat([edge_index, edge_index_reverse], dim=-1)
