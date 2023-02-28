from torch_geometric.utils import add_self_loops, segregate_self_loops, to_networkx, degree
from center_agent import CenterAgent
from single_agent import Agent
from center_env import *
from Utils.eval_helper import *
from Detector.gnn_model import Model
from Detector.data_loader import *
from args import args
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import DataParallel
import torch.nn.functional as F
import torch
import warnings
from copy import deepcopy
import numpy as np
import random as rd
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle as pkl
from collections import Counter
import copy as cp
from tqdm import tqdm
import sys
sys.path.append("../")
warnings.filterwarnings("ignore")

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
rd.seed(args.seed)

dataset = FNNDataset(root="./Detector/data",
                     feature=args.feature, name=args.dataset, train_per=0.2)

data = dataset[0]

with open(f"./Detector/{args.dataset[:3]}_user_news_mapping.pkl", "rb") as file:
    user_news_mapping = pkl.load(file)
with open(f"./Detector/{args.dataset[:3]}_news_user_mapping.pkl", "rb") as file:
    news_user_mapping = pkl.load(file)

user_news_label = {user: sum(int(data.y[i]) for i in news) / len(news)
                   for user, news in user_news_mapping.items()}

bad_users, good_users = [], []
for user, score in user_news_label.items():
    if score > 0.8:
        bad_users.append(user)
    if score < 0.2:
        good_users.append(user)

if args.dataset == "politifact":
    with open("../data/pol_ctrl_acc.pkl", "rb") as f:
        acc_candidate = pkl.load(f)

if args.dataset == "gossipcop":
    with open("../data/gos_ctrl_acc.pkl", "rb") as f:
        acc_candidate = pkl.load(f)

acc_select = defaultdict(list)
acc_select[1] = list(set(acc_candidate[1]) & set(good_users))
acc_select[2] = list(
    set(acc_candidate[4] + acc_candidate[5] + acc_candidate[6]) & set(good_users))
accs = []
for d, acc in acc_candidate.items():
    if d >= 10:
        accs += acc
acc_select[3] = list(set(accs) & set(good_users))

type1 = 100
type2 = 50
type3 = 25

controlled_user = []
controlled_user.append(rd.sample(acc_select[1], type1))
controlled_user.append(rd.sample(acc_select[2], type2))
controlled_user.append(rd.sample(acc_select[3], type3))

target_news = defaultdict(list)
all_news = []

for news, user in news_user_mapping.items():
    all_news.append(news)

    if data.y[news] == 1:
        if len(user) <= 10:
            target_news[1].append(news)
        elif 30 <= len(user) <= 50:
            target_news[2].append(news)
        elif 70 <= len(user) <= 100:
            target_news[3].append(news)

target_news_list = []
target_news_list.append(target_news[1])
target_news_list.append(target_news[2])
target_news_list.append(target_news[3])
target_news_list = [i for j in target_news_list for i in j]

num_news = np.where(1 == data.y.data.numpy())[0][-1] + 1
num_fake_news = np.where(1 == data.y.data.numpy())[0].size

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

surrogate_model = Model(args)
surrogate_model = surrogate_model.to(args.device)
optimizer = torch.optim.Adam(
    surrogate_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if __name__ == "__main__":
    surrogate_model.load_state_dict(
        torch.load(f"./Detector/trained_model/{args.dataset[:3]}_{args.base_model}_hop_{args.nhop}.pth"))

    ctrled_user_list = sum([user for user in controlled_user], [])

    _, _, prob_log, _, _ = eval(surrogate_model, data, target_news_list)

    selected_news = []
    for prob, news in zip(prob_log.tolist(), target_news_list):
        if prob >= 0.5 and prob <= 0.7:
            selected_news.append([news, prob, len(news_user_mapping[news])])
    target_news_list = [i[0] for i in selected_news]

    _, _, old_prob, old_correct, _ = eval(
        surrogate_model, data, target_news_list)

    print(f"Attacking News # {target_news_list}...")

    # num_inject = 5
    # inject_user_list = []
    # for i in range(num_inject):
    #     inject_user_list.append(rd.sample(acc_select[1], 1)[0])
    #     inject_user_list.append(rd.sample(acc_select[2], 1)[0])
    #     inject_user_list.append(rd.sample(acc_select[3], 1)[0])
    #     new_row_1 = torch.mean(data.x[rd.sample(acc_select[1], 20)], dim=0)
    #     new_row_2 = torch.mean(data.x[rd.sample(acc_select[2], 20)], dim=0)
    #     new_row_3 = torch.mean(data.x[rd.sample(acc_select[3], 20)], dim=0)
    #     data.x = torch.cat((torch.tensor(data.x), new_row_1.reshape(1, -1)), axis=0)
    #     data.x = torch.cat((torch.tensor(data.x), new_row_2.reshape(1, -1)), axis=0)
    #     data.x = torch.cat((torch.tensor(data.x), new_row_3.reshape(1, -1)), axis=0)
    #
    # ctrled_user_list = ctrled_user_list + inject_user_list

    # assign new ids to center nodes in sampled subgraph
    all_new_accounts = [i for i in range(len(ctrled_user_list))]

    new_targets = [i for i in range(len(all_new_accounts), len(
        all_new_accounts) + len(target_news_list))]

    center_nodes = ctrled_user_list + new_targets

    center_nodes = torch.tensor(center_nodes, dtype=torch.long)

    # sample two-hop subgraphs
    subgraph_loader = NeighborSampler(data.edge_index.to("cpu"), node_idx=center_nodes,
                                      sizes=[-1, -1], batch_size=int(10e5), num_workers=12)

    for _, n_id, adj in subgraph_loader:
        break

    two_hop_edge, one_hop_edge = adj[0].edge_index, adj[1].edge_index

    # create the new PyG data object for the sampled two-hop subgraph
    new_data = deepcopy(data).to("cpu")
    new_data.num_nodes = n_id.size()[0]
    new_edge_index, _ = add_self_loops(
        torch.cat([one_hop_edge, two_hop_edge], dim=1))

    new_data.edge_index = torch.unique(new_edge_index, dim=1)

    new_data.x = data.x[n_id, :]
    new_data.y = data.y[n_id]

    new_data.two_hop_edge = two_hop_edge
    new_data.one_hop_edge = one_hop_edge
    new_data.to(args.device)

    # random and degree attacks
    # num_edges = 0
    # for target in new_targets:
    #     for node in inject_user_list:
    #         num_edges = num_edges + 1
    #         new_edge = torch.tensor(
    #             [[node, target], [target, node]], dtype=torch.long).cuda()
    #         torch.cat([new_data.edge_index, new_edge], dim=1)

    print(f"\nTotal number of users: {len(ctrled_user_list)}")
    print(f"Type1: {len(ctrled_user_list[0:type1])}")
    print(f"Type2: {len(ctrled_user_list[type1:type1+type2])}")
    print(f"Type3: {len(ctrled_user_list[type1+type2:type1+type2+type3])}")

    num_edges = 0

    for target in rd.sample(new_targets, 1):
        for node in ctrled_user_list[0:type1]:
            num_edges = num_edges + 1
            new_edge = torch.tensor(
                [[node, target], [target, node]], dtype=torch.long).cuda()
            torch.cat([new_data.edge_index, new_edge], dim=1)

    for target in rd.sample(new_targets, 3):
        for node in ctrled_user_list[type1:type1+type2]:
            num_edges = num_edges + 1
            new_edge = torch.tensor(
                [[node, target], [target, node]], dtype=torch.long).cuda()
            torch.cat([new_data.edge_index, new_edge], dim=1)

    for target in rd.sample(new_targets, 5):
        for node in ctrled_user_list[type1+type2:type1+type2+type3]:
            num_edges = num_edges + 1
            new_edge = torch.tensor(
                [[node, target], [target, node]], dtype=torch.long).cuda()
            torch.cat([new_data.edge_index, new_edge], dim=1)

    pred_log, label_log, prob_log, new_correct, loss_test = eval(surrogate_model, new_data.to(args.device),
                                                                 new_targets)

    succ = 0
    for i in range(len(new_correct)):
        if old_correct[i] == 1 and new_correct[i] == 0:
            succ = succ + 1

    succ_rate = succ / len(new_correct)
    print("\033[95msuccess rate: %.2f\033[0m" % (succ_rate))
    print("\033[93mnumber of edges added: %d\033[0m" % num_edges)
