from __future__ import print_function

import os
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
import random
import pickle as pkl
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from q_net_code import QNetNode, NStepQNetNode, node_greedy_actions
from nstep_replay_mem import NstepReplayMem
from args import args


class Agent(object):
    def __init__(self, a_id, env, data, ctrled_accounts, idx_test, idx_train, num_wrong=0):
        self.a_id = a_id
        self.data = data
        self.idx_train = idx_train
        self.idx_test = idx_test
        self.num_wrong = num_wrong
        self.ctrled_accounts = defaultdict(list)
        for t in idx_train + idx_test:
            self.ctrled_accounts[t] = deepcopy(ctrled_accounts)
        self.mem_pool = NstepReplayMem(memory_size=args.mem_size, n_steps=2 * args.num_mod,
                                       balance_sample=args.reward_type == "binary")
        self.env = env

        # self.account_type = account_type

        self.net = QNetNode(self.data, deepcopy(self.ctrled_accounts))
        self.old_net = QNetNode(self.data, deepcopy(self.ctrled_accounts))
        # self.net = NStepQNetNode(2 * args.num_mod, self.data, deepcopy(self.ctrled_accounts))
        # self.old_net = NStepQNetNode(2 * args.num_mod, self.data, deepcopy(self.ctrled_accounts))

        if args.device != "cpu":
            self.net = self.net.to(args.device)
            self.net.data.to(args.device)
            self.old_net = self.old_net.to(args.device)
            self.old_net.data.to(args.device)

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_step = 100000
        self.burn_in = args.burn_in
        self.step = 0
        self.pos = 0
        self.best_eval = None
        self.take_snapshot()

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, time_t, target_list, greedy=False):
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                                      * (self.eps_step - max(0., self.step)) / self.eps_step)

        if random.random() < self.eps and not greedy:
        # if not greedy:
            actions = self.env.uniformRandActions(self.a_id)
            # update the action space
            # for a in actions:
            # 	for t in target_list:
            # 		self.env.list_action_space[t].remove(a)
            # 		self.net.list_action_space[t].remove(a)

            # print(actions)
            # print(sum([self.account_type[a] for a in actions]) / len(target_list))
            # exit()
        else:
            cur_state = self.env.getStateRef()
            with torch.no_grad():
                actions, values = self.net(time_t, cur_state, None, greedy_acts=True)
            actions = list(actions.cpu().numpy())
            # print(actions)
            # print(sum([self.account_type[a] for a in actions]) / len(target_list))
            # exit()

        return actions

    def run_simulation(self):
        if (self.pos + 1) * args.batch_size > len(self.idx_train):
            self.pos = 0
            random.shuffle(self.idx_train)

        selected_idx = self.idx_train[self.pos * args.batch_size: (self.pos + 1) * args.batch_size]
        self.pos += 1
        self.env.setup(self.idx_train)
        self.net.list_action_space = deepcopy(self.ctrled_accounts)

        t = 0
        list_of_list_st = []
        list_of_list_at = []

        while not self.env.isTerminal():
            list_at = self.make_actions(t, self.idx_train)
            list_st = self.env.cloneState()

            # print(list_at, list_st)

            self.env.step(list_at)
            assert (self.env.rewards is not None) == self.env.isTerminal()
            if self.env.isTerminal():
                rewards = self.env.rewards
                s_prime = None
            else:
                rewards = np.zeros(len(list_at), dtype=np.float32)
                s_prime = self.env.cloneState()

            self.mem_pool.add_list(list_st, list_at, rewards, s_prime, [self.env.isTerminal()] * len(list_at), t)
            list_of_list_st.append(deepcopy(list_st))
            list_of_list_at.append(deepcopy(list_at))

            t += 1

        if args.reward_type == "nll":
            return
        T = t
        cands = self.env.sample_pos_rewards(len(selected_idx))
        if len(cands):
            for c in cands:
                sample_idx, target = c
                doable = True
                for t in range(T):
                    if self.ctrled_accounts is not None and (
                    not list_of_list_at[t][sample_idx] in self.ctrled_accounts[target]):
                        doable = False
                        break
                if not doable:
                    continue
                for t in range(T):
                    s_t = list_of_list_st[t][sample_idx]
                    a_t = list_of_list_at[t][sample_idx]
                    s_t = [target, deepcopy(s_t[1])]
                    if t + 1 == T:
                        s_prime = (None, None)
                        r = 1.0
                        term = True
                    else:
                        s_prime = list_of_list_st[t + 1][sample_idx]
                        s_prime = [target, deepcopy(s_prime[1])]
                        r = 0.0
                        term = False
                    self.mem_pool.mem_cells[t].add(s_t, a_t, r, s_prime, term)

    def eval(self):
        self.env.setup(self.idx_test)
        self.net.list_action_space = deepcopy(self.ctrled_accounts)
        t = 0
        while not self.env.isTerminal():
            list_at = self.make_actions(t, self.idx_test, greedy=False)
            print(list_at)
            self.env.step(list_at)
            t += 1

        acc = 1 - (self.env.binary_rewards + 1.0) / 2.0
        acc = np.sum(acc) / (len(self.idx_test) + self.num_wrong)
        print("\033[93m average test: acc %.5f\033[0m" % (acc))

        if args.phase == "train":
            if self.best_eval is None or acc < self.best_eval:
                print("----saving to best attacker since this is the best attack rate so far.----")
                torch.save(self.net.state_dict(), args.save_dir + "/epoch-best.model")
                with open(args.save_dir + "/epoch-best.txt", "w") as f:
                    f.write("%.4f\n" % acc)
                with open(args.save_dir + "/attack_solution.txt", "w") as f:
                    for i in range(len(self.idx_test)):
                        f.write("%d: [" % self.idx_test[i])
                        for e in range(self.env.modified_list[i].added_edges.shape[1]):
                            f.write(f"({self.env.modified_list[i].added_edges[0, e]} {self.env.modified_list[i].added_edges[1, e]})")
                        f.write("] succ: %d\n" % (self.env.binary_rewards[i]))
                self.best_eval = acc

    def train(self):
        pbar = tqdm(range(self.burn_in), unit="batch")
        for p in pbar:
            self.run_simulation()

        # print("-"*10 + f"\ncheck one: {self.env.modified_list[0].modified}\n" + "-"*10 + "\n")

        pbar = range(args.num_steps)
        optimizer = optim.Adam(self.net.parameters(), lr=args.lr)

        self.loss_log = []
        qval_log = []

        for self.step in pbar:

            self.run_simulation()

            # print("-"*10 + f"\ncheck two: {self.env.modified_list[0].modified}\n" + "-"*10 + "\n")

            if self.step % 100 == 0:
                self.take_snapshot()
            if self.step % 200 == 0:
                self.eval()

            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(
                batch_size=args.batch_size)
            list_target = torch.Tensor(list_rt)
            if args.device != "cpu":
                list_target = list_target.to(args.device)

            if not list_term[0]:
                target_nodes, _ = zip(*list_s_primes)
                _, q_t_plus_1 = self.old_net(cur_time + 1, list_s_primes, None)
                _, q_rhs = node_greedy_actions(target_nodes, q_t_plus_1, self.old_net)
                list_target += q_rhs

            # print(list_target.shape)
            list_target = Variable(list_target.view(-1, 1))

            # print(list_target)

            # exit()
            self.net.list_action_space = deepcopy(self.ctrled_accounts)
            _, q_sa = self.net(cur_time, list_st, list_at)
            q_sa = torch.cat(q_sa, dim=0)
            # print(q_sa)
            loss = F.mse_loss(q_sa, list_target)
            optimizer.zero_grad()
            loss.backward()
            # print(self.net.parameters())
            for param in self.net.parameters():
                # print(param)
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            # exit()
            optimizer.step()
            # pbar.set_description("eps: %.5f, loss: %0.5f, q_val: %.5f" % (self.eps, loss.detach().cpu().numpy(),
            # 															  torch.mean(q_sa).detach().cpu().numpy()))
            self.loss_log.append(loss.detach().cpu().numpy())
            qval_log.append(torch.mean(q_sa).detach().cpu().numpy())

        pkl.dump(self.loss_log, open(args.save_dir + "/q_loss.pkl", "wb"))
        pkl.dump(qval_log, open(args.save_dir + "/q_val.pkl", "wb"))
