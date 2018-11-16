import torch
import torch.nn as nn

from .language_model import NextSentencePrediction, MaskedLanguageModel
import pdb

class AdversarialLanguageWrapper(nn.Module):
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
		context_vectors = self.language_model(*args, **kwargs)
		
		# logits for prediction tasks
		token_logits = self.mask_model(context_vectors)
		next_logits = self.next_model(context_vectors)

		# public-private vector similarity loss
		public_vectors, private_vectors = torch.split(context_vectors, self.hidden // 2, -1)
		diff_loss = torch.bmm(private_vectors, torch.transpose(public_vectors, 2, 1))
		diff_loss = torch.sum(diff_loss ** 2) / context_vectors.size(0)

		# adversarial prediction
		language_logits = self.adversary_model(public_vectors)

		return token_logits, next_logits, language_logits, diff_loss


class AdversarialPretrainingWrapper:
	"""
	adds pretraining tasks to entire multilingual model
	"""
	def __init__(self, multilingual_model, adversary_model, hidden, vocab_size):
		self.multilingual_model = multilingual_model
		self.adversary_model = adversary_model
		self.mask_model = MaskedLanguageModel(hidden, vocab_size)
		self.next_model = NextSentencePrediction(hidden)
		# add necessary prediction task
		self.pretraining_models = [AdversarialLanguageWrapper(model, adversary_model, self.mask_model, self.next_model, hidden)
				for model in self.multilingual_model.language_models]

	def __getitem__(self, index):
		if type(index) is str:
			return self.pretraining_models[self.multilingual_model.ltoi[index]]
		else:
			return self.pretraining_models[index]
	
	def __len__(self):
		return len(self.multilingual_model)

	def adversary_forward(self, *args, **kwargs):
		context_vectors = self.multilingual_model.public_model(*args, **kwargs)
		return self.adversary_model(context_vectors)

