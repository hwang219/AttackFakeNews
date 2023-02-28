import numpy as np
import random as rd
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F

from Detector.data_loader import index_to_mask
from args import args


def eval(model, data, target_news_list):
	"""
	Evaluate the attack performance given target news node index
	"""

	target_news_index = torch.tensor(target_news_list, dtype=torch.long)
	data.target_mask = index_to_mask(target_news_index, data.num_nodes)

	out_log = []
	model.eval()
	with torch.no_grad():
		data = data.to(args.device)

		out = model(data)[data.target_mask]
		y = data.y[data.target_mask]
		out_log = [F.softmax(out, dim=1), y]
		loss_test = F.nll_loss(out, y, reduction="none").data.cpu().numpy().tolist()

	pred_log, label_log, prob_log = [], [], []

	pred_y, y = out_log[0].data.cpu().numpy().argmax(
		axis=1), out_log[1].data.cpu().numpy().tolist()
	prob_log.extend(out_log[0].data.cpu().numpy()[:, 1].tolist())
	pred_log.extend(pred_y)
	label_log.extend(y)

	pred_log, label_log, prob_log = np.array(
		pred_log), np.array(label_log), np.array(prob_log)
	correct = (label_log == pred_log).astype(int).tolist()

	return pred_log, label_log, prob_log, correct, loss_test


class ModifiedGraph(object):
	def __init__(self):
		self.added_edges = None
		# self.new_data = deepcopy(data.to("cpu"))
		self.modified = False

	def add_edge(self, x: int, y: int):
		"""
		x: news node
		y: account node
		"""
		new_edges = torch.tensor([[x, y], [y, x]], dtype=torch.long)

		if self.added_edges == None:
			self.added_edges = new_edges
		else:
			self.added_edges = torch.cat([self.added_edges, new_edges], dim=1)
		# self.new_data.edge_index = torch.cat([self.new_data.edge_index, new_edges], dim=1)
		self.modified = True

	def get_new_edges(self):
		# assert self.modified == True
		return self.added_edges


class NodeAttakEnv(object):
	def __init__(self, data, ctrled_accounts, all_targets, classifier):
		self.classifier = classifier
		self.all_targets = all_targets
		self.static_data = data
		self.agent_ctrled_accounts = defaultdict(dict)
		self.all_ctrled_accounts = defaultdict(list)
		for t in all_targets:
			for accounts in ctrled_accounts.values():
				self.all_ctrled_accounts[t] += deepcopy(accounts)
		for a_id in ctrled_accounts:
			agent_ctrled_accounts = defaultdict(list)
			for t in all_targets:
				agent_ctrled_accounts[t] = deepcopy(ctrled_accounts[a_id])
			self.agent_ctrled_accounts[a_id] = agent_ctrled_accounts

	def setup(self, target_nodes):
		self.target_nodes = target_nodes
		self.n_steps = 0
		self.rewards = None
		self.binary_rewards = None
		self.modified_list = []
		self.list_action_space = deepcopy(self.agent_ctrled_accounts)

		for i in range(len(self.target_nodes)):
			self.modified_list.append(ModifiedGraph())

		self.list_acc_of_all = []

	def step(self, actions, eval_flag=False):

		# num_inject = 300
		# inject_user_list = []
		# for i in range(num_inject):
		#     inject_user_list.append(rd.randint(0, 175))
		#
		# actions = actions + inject_user_list

		# inject_user = rd.sample(list(self.agent_ctrled_accounts[2].values())[0], 1)
		# actions.append(inject_user)
		# new_row = self.static_data.x[inject_user]
		# self.static_data.x = torch.cat((torch.tensor(self.static_data.x), new_row.reshape(1, -1)), axis=0)

		for i in range(len(self.target_nodes)):
			# assert self.first_nodes[i] != actions[i]

			#print(self.target_nodes[i], actions[i])
			self.modified_list[i].add_edge(self.target_nodes[i], actions[i])

			#print(self.modified_list[i].added_edges)

		self.banned_list = None
		self.n_steps += 1

		# if self.isTerminal():
		# 	acc_list = []
		# 	loss_list = []
		# 	for i in tqdm(range(len(self.target_nodes))):
		# 		new_edges = self.modified_list[i].get_new_edges()
		# 		temp_new_data = deepcopy(self.static_data).to("cpu")
		# 		temp_new_data.edge_index = torch.cat([temp_new_data.edge_index, new_edges], dim=1)
		# 		pred_log, label_log, prob_log, correct, loss = eval(self.classifier, temp_new_data.to(args.device), self.target_nodes)
		# 		# cur_idx = self.all_targets.index(self.target_nodes[i])
		# 		self.list_acc_of_all.append(np.array(correct))
		# 		acc_list.append(correct[i])
		# 		loss_list.append(loss[i])
		# 	self.binary_rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
		# 	if args.reward_type == "binary":
		# 		self.rewards = (np.array(acc_list) * -2.0 + 1.0).astype(np.float32)
		# 	else:
		# 		assert args.reward_type == "nll"
		# 		self.rewards = np.array(loss_list).astype(np.float32)

		# evaluate all actions under the same subgraph
		if self.isTerminal():
			acc_list = []
			loss_list = []
			temp_new_data = deepcopy(self.static_data).to("cpu")

			# for i in tqdm(range(len(self.target_nodes))):

			for i in range(len(self.target_nodes)):
				new_edges = self.modified_list[i].get_new_edges()

				#print(temp_new_data.edge_index)
				#temp_new_data.edge_index = torch.cat([temp_new_data.edge_index, new_edges], dim=1)
				#temp_new_data.edge_index = torch.cat([temp_new_data.edge_index, new_edges], dim=1)

				idx = temp_new_data.edge_index[0][(temp_new_data.edge_index[0] != self.target_nodes[i]) & (
					temp_new_data.edge_index[0] != actions[i])]
				temp_new_data.edge_index = temp_new_data.edge_index[:, idx]

				#print(temp_new_data.edge_index)
			pred_log, label_log, prob_log, correct, loss = eval(
				self.classifier, temp_new_data.to(args.device), self.target_nodes)

			if eval_flag:
				print(f"new prob: {prob_log[0]:.4f}")
			# cur_idx = self.all_targets.index(self.target_nodes[i])
			self.list_acc_of_all.append(np.array(correct))
			self.binary_rewards = (np.array(correct) * -2.0 + 1.0).astype(np.float32)

			if args.reward_type == "binary":
				self.rewards = (np.array(correct) * -2.0 + 1.0).astype(np.float32)
			else:
				assert args.reward_type == "nll"
				self.rewards = np.array(loss).astype(np.float32)
			#print(self.rewards)

	def sample_pos_rewards(self, num_samples):
		assert self.list_acc_of_all is not None
		cands = []
		for i in range(len(self.list_acc_of_all)):
			succ = np.where(self.list_acc_of_all[i] < 0.9)[0]
			for j in range(len(succ)):
				cands.append((i, self.all_targets[succ[j]]))
		if num_samples > len(cands):
			return cands
		rd.shuffle(cands)
		return cands[0:num_samples]

	def uniformRandActions(self, a_id):
		act_list = []
		offset = 0
		for i in range(len(self.target_nodes)):
			cur_node = self.target_nodes[i]
			region = self.list_action_space[a_id][cur_node]
			cur_action = region[np.random.randint(len(region))]
			act_list.append(cur_action)
		return act_list

	def isTerminal(self):
		if self.n_steps == 2 * args.num_mod:
			return True
		return False

	def getStateRef(self):

		return list(zip(self.target_nodes, self.modified_list))

	def cloneState(self):

		return list(zip(self.target_nodes[:], deepcopy(self.modified_list)))
