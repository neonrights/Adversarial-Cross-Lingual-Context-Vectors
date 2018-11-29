import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageTranslator(nn.Module):
	def __init__(self, language_model, translator_model, target_language_model):
		self.language_model = language_model
		self.translator_model = translator_model
		self.target_language_model = target_language_model

	def forward(self, input_ids, target_ids, input_mask=None, target_mask=None):
		language_vectors, _ = self.language_model(input_ids, attention_mask=input_mask)
		target_vectors, _ = self.target_language_model(target_ids, attention_mask=target_mask)
		logits = self.translator_model(language_vectors[-1], target_vectors[-1], input_mask, target_mask)


class MultilingualTranslator:
	def __init__(self, multilingual_model, translator_model, target_language):
		self.ltoi = multilingual_model.ltoi
		self.models = multilingual_model.models.copy()
		self.models['translator'] = translator_model
		self.models['target_language'] = multilingual_model[target_language]
		self.language_translators = [LanguageTranslator(model, translator_model, self.models['target_language'])
				for model in multilingual_model.language_models]

	def __getitem__(self, index):
		if type(index) is str:
			return self.language_models[self.ltoi[index]]
		else:
			return self.language_models[index]
	
	def __len__(self):
		return len(self.ltoi)

	def get_components(self):
		return self.models

