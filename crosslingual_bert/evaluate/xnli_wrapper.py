import torch
import torch.nn as nn


class MixtureOfExperts(nn.Module):
	def __init__(self, input_dim, output_dim, expert_count):
		self.linear = nn.Linear(input_dim, output_dim * expert_count)
		self.expert_gate = nn.Linear(input_dim, expert_count)
		self.output_dim = output_dim
		self.expert_count = expert_count

	def forward(self, input_vectors):
		logits = self.linear(input_vectors).reshape(-1, self.output_dim, self.input_dim)
		activations = nn.Softmax(dim=-1)(self.expert_gate(input_vectors)).unsqueeze(-1)
		return torch.bmm(logits, activations).squeeze()


class TranslatorWrapper(nn.Module):
	def __init__(self, multilingual_model, translator_model, hidden, vocab_size):
		self.models = multilingual_model.models.copy()
		self.ltoi = multilingual_model.ltoi
		# machine translation model
		self.translator_model = translator_model
		self.models['MoE'] = MixtureOfExperts(hidden, vocab_size, len(self.ltoi))
		self.moe_models = [TranslatorLanguageWrapper(model, self.translator_model) for model in ]

	def __getitem__(self, index):
		if type(index) is str:
			return self.moe_models[self.ltoi[index]]
		else:
			return self.moe_models[index]

	def get_components(self):
		return self.models


class TranslatorLanguageWrapper(nn.Module):
	def __init__(self, language_model, translator_model, moe_model):
		# weights needed for softmax
		self.language_model = language_model
		self.translator_model = translator_model
		self.moe_model = moe_model

	def forward(self, input_ids, output_ids, input_mask=None, output_mask=None):
		in_hidden_vectors, in_pooled_vectors = self.language_model(input_ids, attention_mask=input_mask)
		out_hidden_vectors, out_pooled_vectors = self.translator_model(input_hidden_vectors[-1], output_ids, output_mask)
		return self.moe_model(out_pooled_vectors)
