import torch
import torch.nn as nn

from .language_model import NextSentencePrediction, MaskedLanguageModel
import pdb

class SingleBERTWrapper(nn.Module):
	"""
	wrapper that adds pretraining prediction for a single language model
	"""
	def __init__(self, language_model, adversary_model, mask_model, next_model, hidden):
		super().__init__()
		self.language_model = language_model
		self.adversary_model = adversary_model
		self.mask_model = mask_model
		self.next_model = next_model
		self.hidden = hidden
	
	def forward(self, *args, **kwargs):
		hidden_vectors, pooled_vectors = self.language_model(*args, **kwargs)
		
		# logits for prediction tasks
		token_logits = self.mask_model(hidden_vectors[-1])
		next_logits = self.next_model(pooled_vectors)

		# public-private vector similarity loss
		public_vectors, private_vectors = torch.split(hidden_vectors[-1], self.hidden // 2, -1)
		diff = torch.bmm(private_vectors, torch.transpose(public_vectors, 1, 2))
		diff_loss = torch.sum(diff ** 2)
		diff_loss /= pooled_vectors.size(0)

		# adversarial prediction
		public_pooled, _ = torch.split(pooled_vectors, self.hidden // 2, -1)
		language_logits = self.adversary_model(public_pooled)

		return token_logits, next_logits, language_logits, diff_loss


class AdversarialBERTWrapper:
	"""
	adds pretraining tasks to entire multilingual model
	"""
	def __init__(self, multilingual_model, adversary_model, hidden, vocab_size):
		self.multilingual_model = multilingual_model
		self.adversary_model = adversary_model
		self.mask_model = MaskedLanguageModel(hidden, vocab_size)
		self.next_model = NextSentencePrediction(hidden)
		# add necessary prediction task
		self.pretraining_models = [SingleBERTWrapper(model, adversary_model, self.mask_model, self.next_model, hidden)
				for model in self.multilingual_model.language_models]

	def __getitem__(self, index):
		if type(index) is str:
			return self.pretraining_models[self.multilingual_model.ltoi[index]]
		else:
			return self.pretraining_models[index]
	
	def __len__(self):
		return len(self.multilingual_model)

	def adversary_forward(self, *args, **kwargs):
		_, pooled_vectors = self.multilingual_model.public_model(*args, **kwargs)
		return self.adversary_model(pooled_vectors)

	def get_components(self):
		components = self.multilingual_model.get_components()
		components['adversary'] = self.adversary_model
		components['mask_prediction'] = self.mask_model
		components['is_next_prediction'] = self.next_model
		return components		

