import os.path as osp

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import remove_self_loops, to_undirected, add_self_loops
from torch_geometric.io import read_txt_array
from torch_sparse import coalesce

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def read_graph_data(folder, feature, train_per):

    node_attributes = np.load(folder + f'{feature}_feature.npy')
    graph = loadmat(folder + 'retweet_graph')
    adj = graph['graph']
    y = graph['label']

    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    x = torch.from_numpy(node_attributes).to(torch.float)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = add_self_loops(edge_index, None)

    edge_index, edge_attr = coalesce(
        edge_index, edge_attr, num_nodes, num_nodes)

    X_train, X_test, y_train, y_test = train_test_split(range(y.size), y.squeeze(),
                                                        test_size=1 - train_per - 0.1, random_state=777)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=train_per / (train_per + 0.1), random_state=777)

    train_index = torch.tensor(X_train, dtype=torch.long)
    val_index = torch.tensor(X_val, dtype=torch.long)
    test_index = torch.tensor(X_test, dtype=torch.long)

    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(val_index, size=num_nodes)
    test_mask = index_to_mask(test_index, size=num_nodes)

    y = torch.from_numpy(y.squeeze()).to(torch.long)
    # padding the label for user nodes
    y = torch.cat((y, torch.zeros(num_nodes - y.size()[0], dtype=torch.long)))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


class ToUndirected:
    def __init__(self):
        """
        Transform the graph to the undirected graph
        """
        pass

    def __call__(self, data):
        edge_attr = None
        edge_index = to_undirected(data.edge_index, data.x.size(0))
        num_nodes = edge_index.max().item() + 1 if data.x is None else data.x.size(0)
        # edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = coalesce(
            edge_index, edge_attr, num_nodes, num_nodes)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data


class FNNDataset(InMemoryDataset):
    r"""
            The Graph datasets built upon FakeNewsNet data

    Args:
            root (string): Root directory where the dataset should be saved.
            name (string): The `name
                    <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
                    dataset.
            transform (callable, optional): A function/transform that takes in an
                    :obj:`torch_geometric.data.Data` object and returns a transformed
                    version. The data object will be transformed before every access.
                    (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                    an :obj:`torch_geometric.data.Data` object and returns a
                    transformed version. The data object will be transformed before
                    being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                    :obj:`torch_geometric.data.Data` object and returns a boolean
                    value, indicating whether the data object should be included in the
                    final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, name, feature='glove', train_per=0.2, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        self.feature = feature
        self.train_per = train_per
        super(FNNDataset, self).__init__(
            root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw/'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed/'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def raw_file_names(self):
        return [f'{self.feature}_feature.npy', 'retweet_graph.mat']

    @property
    def processed_file_names(self):
        if self.pre_filter is None:
            return f'{self.name[:3]}_data_{self.feature}.pt'
        else:
            return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

    def process(self):

        data = read_graph_data(self.raw_dir, self.feature, self.train_per)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
