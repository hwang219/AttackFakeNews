import argparse
import time
from tqdm import tqdm
import copy as cp
from collections import Counter
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict
import random as rd

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DataParallel
from torch_geometric.data import DataLoader, DataListLoader

from data_loader import *
from gnn_model import Model
from eval_helper import *


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.model = args.base_model
        self.num_layers = args.nhop

        if self.model == 'gcn':
            if self.num_layers == 1:
                self.conv1 = GCNConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.conv1 = GCNConv(self.num_features, self.nhid * 2)
                self.conv2 = GCNConv(self.nhid * 2, self.nhid)
        elif self.model == 'sage':
            if self.num_layers == 1:
                self.conv1 = SAGEConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.conv1 = SAGEConv(self.num_features, self.nhid * 2)
                self.conv2 = SAGEConv(self.nhid * 2, self.nhid)
        elif self.model == 'gat':
            if self.num_layers == 1:
                self.conv1 = GATConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.conv1 = GATConv(self.num_features, self.nhid * 2)
                self.conv2 = GATConv(self.nhid * 2, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        if self.num_layers == 2:
            x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.log_softmax(self.lin1(x), dim=-1)

        return x


def compute_test(data, test=True):
    model.eval()
    with torch.no_grad():
        if not args.multi_gpu:
            data = data.to(args.device)
        out = model(data)
        if test:
            out = out[data.test_mask]
        else:
            out = out[data.val_mask]

        if args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0)
                          for d in data]).squeeze().to(out.device)
        else:
            if test:
                y = data.y[data.test_mask]
            else:
                y = data.y[data.val_mask]
        out_log = [F.softmax(out, dim=1), y]
        loss_test = F.nll_loss(out, y).item()
    return eval_hin(out_log), loss_test


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str,
                    default='politifact', help='[politifact, gossipcop, MMCOVID]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float,
                    default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--nhop', type=int, default=2, help='number of hops')
parser.add_argument('--dropout_ratio', type=float,
                    default=0.0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--no_feature', type=bool, default=False,
                    help='whether including node feature')
parser.add_argument('--multi_gpu', type=bool,
                    default=False, help='multi-gpu mode')
parser.add_argument('--self_loop', type=bool, default=False,
                    help='whether only keeping the self loops')
parser.add_argument('--feature', type=str, default='glove',
                    help='feature type, [hand, tfidf, glove, bert]')
parser.add_argument('--base_model', type=str, default='gat',
                    help='model type, [gcn, gat, sage]')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
rd.seed(args.seed)

dataset = FNNDataset(root='data',
                     feature=args.feature, name=args.dataset, train_per=0.2)

data = dataset[0]

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

if args.self_loop:
    mask = data.edge_index[0, :] == data.edge_index[1, :]
    data.edge_index = torch.masked_select(data.edge_index, mask).reshape(2, -1)


model = Model(args)
if args.multi_gpu:
    model = DataParallel(model)
model = model.to(args.device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if __name__ == '__main__':
    out_log = []
    model.train()
    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        correct = 0
        optimizer.zero_grad()
        if not args.multi_gpu:
            data = data.to(args.device)
        out = model(data)[data.train_mask]
        if args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0)
                          for d in data]).squeeze().to(out.device)
        else:
            y = data.y[data.train_mask]

        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        out_log = [F.softmax(out, dim=1), y]
        acc_train, _, _, _, recall_train, auc_train, _ = eval_hin(out_log)
        [acc_val, _, _, _, recall_val, auc_val,
            _], loss_val = compute_test(data, test=False)
        print(
            f"Epoch: {epoch+1} | "
            f"loss_train: {loss_train:.4f} | "
            f"acc_train: {acc_train:.4f} | "
            f"recall_train: {recall_train:.4f} | "
            f"auc_train: {auc_train:.4f} | "
            f"loss_val: {loss_val:.4f} | "
            f"acc_val: {acc_val:.4f} | "
            f"recall_val: {loss_val:.4f} | "
            f"auc_val: {auc_val:.4f}"
        )

    [acc, f1_macro, f1_micro, precision, recall,
        auc, ap], test_loss = compute_test(data)
    print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
          f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')

    torch.save(model.state_dict(
    ), f"trained_model/{args.dataset[:3]}_{args.base_model}_hop_{args.nhop}.pth")
