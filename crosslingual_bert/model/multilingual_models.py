import copy
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_official import BertConfig, BERTEmbeddings, NoEmbedBert
from .translator import TranslatorModel


class MultilingualConfig(BertConfig):
	def __init__(self,
				vocab_size,
				languages,
				hidden_size=768,
				num_hidden_layers=12,
				num_attention_heads=12,
				intermediate_size=3072,
				hidden_act="gelu",
				hidden_dropout_prob=0.1,
				attention_probs_dropout_prob=0.1,
				max_position_embeddings=512,
				type_vocab_size=16,
				initializer_range=0.02,
				checkpoint_layers=False):
		self.languages = languages
		super().__init__(vocab_size,
				hidden_size,
				num_hidden_layers,
				num_attention_heads,
				intermediate_size,
				hidden_act,
				hidden_dropout_prob,
				attention_probs_dropout_prob,
				max_position_embeddings,
				type_vocab_size,
				initializer_range,
				checkpoint_layers)

	@classmethod
	def from_dict(cls, json_object):
		config = MultilingualBertConfig(vocab_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config


class MultilingualBert(nn.Module):
	"""Cross-lingual context vector model using transformer architecture
	"""
	def __init__(self, config: MultilingualConfig):
		super().__init__()
		self.shared = NoEmbedBert(config)
		self.embeddings = BERTEmbeddings(config)
		self.private = {language: NoEmbedBert(config) for language in config.languages}
		for language, model in self.private.items():
			self.add_module(language, model)

	def forward(self, language, input_ids, token_type_ids=None, attention_mask=None):
		assert language in self.private
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		# embeddings shared across all languages
		embeddings = self.embeddings(input_ids, token_type_ids)

		shared_vectors, shared_pooled = self.shared(embeddings, attention_mask)
		private_vectors, private_pooled = self.private[language](embeddings, attention_mask)
		hidden_vectors = [torch.cat((sv, pv), -1) for sv, pv in zip(shared_vectors, private_vectors)]
		pooled_output = torch.cat((shared_pooled, private_pooled), -1)
		return hidden_vectors, pooled_output

	def language_parameters(self, language):
		"""Returns all parameters for a specific language model
		"""
		assert language in self.private
		return chain(self.shared.parameters(),
			self.private[language].parameters(),
			self.embeddings.parameters())


class MultilingualTranslator(nn.Module):
	"""Universal to target language translation model using transformer architecture
	"""
	def __init__(self, model: MultilingualBert, target_language: str, config: MultilingualConfig):
		assert target_language in model.private
		super().__init__()
		self.multilingual_model = model
		
		# double intermediate and hidden size to account for shared-private model
		config = copy.copy(config)
		config.intermediate_size *= 2
		config.hidden_size *= 2

		self.translator_model = TranslatorModel(config)
		self.target_language = target_language

	def forward(self, language, input_ids, target_ids, input_mask=None, target_mask=None):
		language_vectors, _ = self.multilingual_model(language, input_ids, attention_mask=input_mask)
		target_vectors, _ = self.multilingual_model(self.target_language, target_ids, attention_mask=target_mask)
		return self.translator_model(language_vectors[-1], target_vectors[-1], input_mask, target_mask)

	def language_parameters(self, language):
		return chain(self.translator_model.parameters(), self.multilingual_model.language_parameters(language))

