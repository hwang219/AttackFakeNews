from __future__ import print_function

import os
import sys
import numpy as np
from copy import deepcopy

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling, SAGEConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from args import args


def node_greedy_actions(target_nodes, list_q, net):
	assert len(target_nodes) == len(list_q)

	actions = []
	values = []

	selected_actions = []

	for i in range(len(target_nodes)):
		region = net.list_action_space[target_nodes[i]]

		if region is None:
			assert list_q[i].size()[0] == net.total_nodes
		else:
			assert len(region) == list_q[i].size()[0]

		vals, act_indices = list_q[i].sort(descending=True, dim=0)

		# avoid repeated account selection for multiple actions
		for val, act_index in zip(vals, act_indices):
			# val, act_index = torch.max(list_q[i], dim=0)
			if act_index not in net.all_selected_actions:
				net.all_selected_actions.append(act_index)
				break

		values.append(val)
		# print(act_index)
		net.list_q = list_q

		# print(list_q)
		# print(list_q[i])
		# print(act_index)
		act = region[act_index.data.cpu().numpy()[0]]

		act = Variable(torch.LongTensor([act]))
		actions.append(act)

	return torch.cat(actions, dim=0).data, torch.cat(values, dim=0).data


class QNetNode(nn.Module):
	def __init__(self, data, list_action_space):
		super(QNetNode, self).__init__()
		self.data = data
		self.list_action_space = list_action_space
		self.all_selected_actions = []

		embed_dim = args.latent_dim
		if args.bilin_q:
			last_wout = embed_dim
		else:
			last_wout = 1
			self.bias_target = Parameter(torch.Tensor(1, embed_dim))

		if args.mlp_hidden:
			self.linear_1 = nn.Linear(embed_dim, args.mlp_hidden)
			self.linear_out = nn.Linear(args.mlp_hidden, last_wout)
		else:
			self.linear_out = nn.Linear(embed_dim, last_wout)

		self.conv1 = SAGEConv(self.data.num_features, embed_dim)
		# self.pool1 = TopKPooling(embed_dim, ratio=0.8)
		self.conv2 = SAGEConv(embed_dim, embed_dim)
		# self.pool2 = TopKPooling(embed_dim, ratio=0.8)

		# self.lin1 = torch.nn.Linear(256, 128)
		# self.lin2 = torch.nn.Linear(128, 64)

	def forward(self, time_t, states, actions, greedy_acts=False):

		input_node_linear = F.relu(self.conv1(self.data.x, self.data.edge_index))
		# input_node_linear, edge_index, _, batch, _, _ = self.pool1(input_node_linear, self.data.edge_index, None, None)
		# input_node_linear = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

		target_nodes, batch_graph = zip(*states)

		# print(f"batch graph length is {len(batch_graph)}")

		list_pred = []
		prefix_sum = []
		for i in range(len(batch_graph)):
			# print(i)

			region = self.list_action_space[target_nodes[i]]

			node_embed = input_node_linear.clone()
			new_edges = batch_graph[i].get_new_edges()
			# print(new_edges)
			# print(i)

			# some new edges have been added to the graph
			if new_edges is not None:
				# print(new_edges)
				node_embed = F.relu(self.conv2(node_embed, new_edges.to(args.device)))
				# node_embed, edge_index, _, batch, _, _ = self.pool2(node_embed, new_edges, None, None)

			if not args.bilin_q:
				node_embed[target_nodes[i]] += self.bias_target[0]

			# node_embed = F.relu(self.lin1(node_embed))

			target_embed = node_embed[target_nodes[i], :].view(-1, 1)
			if region is not None:
				node_embed = node_embed[region]

			# graph_embed = torch.mean(node_embed, dim=0, keepdim=True)

			if actions is None:
				pass
				# graph_embed = graph_embed.repeat(node_embed.size()[0], 1)
			else:
				if region is not None:
					act_idx = region.index(actions[i])
				else:
					act_idx = actions[i]
				node_embed = node_embed[act_idx, :].view(1, -1)

			# embed_s_a = torch.cat((node_embed, graph_embed), dim=1)
			embed_s_a = node_embed

			if args.mlp_hidden:
				embed_s_a = F.relu(self.linear_1(embed_s_a))
			raw_pred = self.linear_out(embed_s_a)

			if args.bilin_q:
				raw_pred = torch.mm(raw_pred, target_embed)
			list_pred.append(raw_pred)

		if greedy_acts:
			# sizes = [q.size() for q in list_pred]
			# print(sizes)
			actions, _ = node_greedy_actions(target_nodes, list_pred, self)

		return actions, list_pred


class NStepQNetNode(nn.Module):
	def __init__(self, num_steps, data, list_action_space):
		super(NStepQNetNode, self).__init__()
		self.data = data
		self.list_action_space = list_action_space


		list_mod = []

		for i in range(0, num_steps):
			list_mod.append(QNetNode(data, list_action_space))

		self.list_mod = nn.ModuleList(list_mod)

		self.num_steps = num_steps

	def forward(self, time_t, states, actions, greedy_acts=False, is_inference=False):
		assert time_t >= 0 and time_t < self.num_steps

		return self.list_mod[time_t](time_t, states, actions, greedy_acts, is_inference)
