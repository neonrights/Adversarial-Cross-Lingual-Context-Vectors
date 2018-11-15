import torch.nn as nn


class DummyAdversary(nn.Module):
	def __init__(self, hidden_size, language_size):
		super().__init__()
		self.linear = nn.Linear(hidden_size, language_size)

	def forward(self, inputs):
		return self.linear(inputs[:, 0])