import torch.nn as nn

class BaseNetwork(nn.Module):
	def __init__(self, num_features, num_output):
		super().__init__()

		self.num_features = num_features
		self.num_output = num_output

	def forward(self, x, cv):
		raise NotImplementedError()