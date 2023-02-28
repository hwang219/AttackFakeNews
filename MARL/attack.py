from center_agent import CenterAgent
from single_agent import Agent
from center_env import *
from Utils.eval_helper import *
from Detector.gnn_model import Model
from Detector.data_loader import *
from args import args
from torch_geometric.utils import add_self_loops, to_networkx
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
acc_select[1] = acc_candidate[1]
acc_select[2] = acc_candidate[4] + acc_candidate[5] + acc_candidate[6]
accs = []
for d, acc in acc_candidate.items():
    if d >= 10:
        accs += acc
acc_select[3] = accs

controlled_user = []
controlled_user.append(rd.sample(acc_select[1], 100))
controlled_user.append(rd.sample(acc_select[2], 50))
controlled_user.append(rd.sample(acc_select[3], 20))

target_news = defaultdict(list)
for news, user in news_user_mapping.items():
    # 1 represent fake news
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

    pred_log, old_label_log, prob_log, old_correct, loss_test = eval(surrogate_model, data,
                                                                     target_news_list)

    selected_news = []
    for prob, news in zip(prob_log.tolist(), target_news_list):
        # if prob >= 0.5 and prob <= 0.7:
        selected_news.append([news, prob, len(news_user_mapping[news])])
    target_news_list = [i[0] for i in selected_news]

    # num_inject = 20
    # inject_user_list = []
    # for i in range(num_inject):
    #     inject_user_list.append(rd.sample(acc_select[3], 1)[0])
    #     new_row = torch.mean(data.x[rd.sample(acc_select[3], 5)], dim=0)
    #     data.x = torch.cat(
    #         (torch.tensor(data.x), new_row.reshape(1, -1)), axis=0)
    # ctrled_user_list = ctrled_user_list + inject_user_list

    succ_list = []
    news_attack_succ = []

    print(f"Attacking News # {target_news_list}...")
    for idx, target_news in enumerate(target_news_list):

        print(
            f"Attacking News # {target_news} with old probability: {selected_news[idx][1]:.4f} ")

        all_new_accounts = [i for i in range(len(ctrled_user_list))]

        agent_ctrl_accounts = dict()
        start_idx = 0
        end_idx = 0
        for a_id, user in enumerate(controlled_user):
            end_idx = start_idx + len(user)
            agent_ctrl_accounts[a_id] = all_new_accounts[start_idx:end_idx]
            start_idx += len(user)

        new_target = len(all_new_accounts)

        center_nodes = ctrled_user_list + [target_news]
        center_nodes = torch.tensor(center_nodes, dtype=torch.long)

        subgraph_loader = NeighborSampler(data.edge_index.to("cpu"), node_idx=center_nodes,
                                          sizes=[-1, -1], batch_size=int(10e5), num_workers=12)

        for _, n_id, adj in subgraph_loader:
            break

        two_hop_edge, one_hop_edge = adj[0].edge_index, adj[1].edge_index

        new_data = deepcopy(data).to("cpu")
        new_data.num_nodes = n_id.size()[0]
        new_edge_index, _ = add_self_loops(
            torch.cat([one_hop_edge, two_hop_edge], dim=1))
        new_data.edge_index = torch.unique(new_edge_index, dim=1)
        new_data.x = data.x[n_id, :]
        new_data.y = data.y[n_id]
        new_data.two_hop_edge = two_hop_edge
        new_data.one_hop_edge = one_hop_edge

        pred_log, label_log, prob_log, correct, loss_test = eval(surrogate_model, new_data.to(args.device),
                                                                 [new_target])

        subgraph_loader = NeighborSampler(data.edge_index.to("cpu"), node_idx=center_nodes,
                                          sizes=[50, 50], batch_size=int(10e5), num_workers=12)

        for _, n_id, adj in subgraph_loader:
            break

        two_hop_edge, one_hop_edge = adj[0].edge_index, adj[1].edge_index

        q_net_data = deepcopy(data).to("cpu")
        q_net_data.num_nodes = n_id.size()[0]
        new_edge_index, _ = add_self_loops(
            torch.cat([one_hop_edge, two_hop_edge], dim=1))
        q_net_data.edge_index = torch.unique(new_edge_index, dim=1)
        q_net_data.x = data.x[n_id, :]
        q_net_data.y = data.y[n_id]
        q_net_data.two_hop_edge = two_hop_edge
        q_net_data.one_hop_edge = one_hop_edge

        meta_list, attack_list = [new_target], [new_target]

        env = NodeAttakEnv(new_data, agent_ctrl_accounts,
                           [new_target], surrogate_model)

        agents = dict()
        for a_id, accounts in agent_ctrl_accounts.items():
            agents[a_id] = Agent(a_id, env, q_net_data, accounts, meta_list, attack_list,
                                 num_wrong=len(correct) - sum(correct))

        center_agent = CenterAgent(env, q_net_data, agent_ctrl_accounts, meta_list, attack_list, agents,
                                   num_wrong=len(correct) - sum(correct))
        center_agent.train()

        if center_agent.env.binary_rewards[0] == 1:
            news_attack_succ.append(target_news)

        succ_list.append(center_agent.env.binary_rewards[0])

succ_rate = len([i for i in succ_list if i == 1.0]) / len(target_news_list)
print("\033[95msuccess rate: succ_rate %.2f\033[0m" % (succ_rate))
print(news_attack_succ)
