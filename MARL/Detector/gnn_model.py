import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

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