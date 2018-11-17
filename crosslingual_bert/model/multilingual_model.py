import torch
import torch.nn as nn
import torch.nn.functional as F


class PublicPrivateModel(nn.Module):
	"""
	shared context vector model for a single language
	"""
	def __init__(self, public, private):
		super().__init__()
		self.public = public
		self.private = private
	
	def forward(self, *args, **kwargs):
		return torch.cat([self.public(*args, **kwargs), self.private(*args, **kwargs)], -1)


class MultilingualModel:
	"""
	Entire cross-lingual context vector model.
	Supports indexing by language name or label to return specific language model.
	"""
	def __init__(self, language_ids, arch, *arch_args, **arch_kwargs):
		self.ltoi = language_ids # language to index/label id
		self.public_model = arch(*arch_args, **arch_kwargs)
		self.private_models = [arch(*arch_args, **arch_kwargs) for _ in range(len(language_ids))]
		self.language_models = [PublicPrivateModel(self.public_model, private_model) for private_model in self.private_models]
	
	def __getitem__(self, index):
		if type(index) is str:
			return self.language_models[self.ltoi[index]]
		else:
			return self.language_models[index]
	
	def __len__(self):
		return len(self.ltoi)

	def parameters(self):
		return [self.public_model] + self.private_models

