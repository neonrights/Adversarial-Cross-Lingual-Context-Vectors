import torch.nn as nn


class DummyAdversary(nn.Module):
	def __init__(self, hidden_size, language_size):
		super().__init__()
		self.linear = nn.Linear(hidden_size, language_size)
		self.softmax = nn.LogSoftmax(dim=-1)

	def forward(self, inputs):
		return self.softmax(self.linear(inputs[:, 0]))