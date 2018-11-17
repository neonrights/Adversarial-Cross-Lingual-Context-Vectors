import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path


class PublicPrivateBERT(nn.Module):
	"""
	shared context vector model for a single language
	"""
	def __init__(self, public, private):
		super().__init__()
		self.public = public
		self.private = private
	
	def forward(self, *args, **kwargs):
		public_hidden, public_pooled = self.public(*args, **kwargs)
		private_hidden, private_pooled = self.private(*args, **kwargs)
		hidden_vectors = [torch.cat([public, private], -1) for public, private in zip(public_hidden, private_hidden)]
		pooled_vectors = torch.cat([public_pooled, private_pooled], -1)
		return hidden_vectors, pooled_vectors


class MultilingualBERT:
	"""
	Entire cross-lingual context vector model.
	Supports indexing by language name or label to return specific language model.
	"""
	def __init__(self, language_ids, arch, *arch_args, **arch_kwargs):
		self.ltoi = language_ids # language to index/label id
		self.public_model = arch(*arch_args, **arch_kwargs)
		self.private_models = [arch(*arch_args, **arch_kwargs) for _ in range(len(language_ids))]
		self.language_models = [PublicPrivateBERT(self.public_model, private_model) for private_model in self.private_models]
	
	def __getitem__(self, index):
		if type(index) is str:
			return self.language_models[self.ltoi[index]]
		else:
			return self.language_models[index]
	
	def __len__(self):
		return len(self.ltoi)

	def get_components(self):
		# returns all sub-models and names
		components = {language: self.private_models[index] for language, index in self.ltoi.items()}
		components['public'] = self.public_model
		return components
