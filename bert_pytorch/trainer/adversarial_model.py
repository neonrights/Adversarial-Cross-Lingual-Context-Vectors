import torch
import torch.nn as nn


class AdversarialLanguageWrapper(nn.Module):
	"""
	wrapper that adds pretraining prediction for a single language model
	"""
	def __init__(self, model, adversary, hidden, vocab_size): 
		super().__init__()
		self.model = model
		self.adversary = adversary
		self.hidden = hidden

		# variables necessary for prediction tasks
		self.token_linear = nn.Linear(hidden, vocab_size)
		self.next_linear = nn.Linear(hidden, 2)
	
	def forward(self, *args, **kwargs):
		context_vectors = self.model(*args, **kwargs)
		
		# logits for prediction tasks
		token_logits = self.token_linear(context_vectors)
		next_logits = self.next_linear(context_vectors[:, 0])
		language_logits = self.adversary(context_vectors)

		# public-private vector similarity loss
		public_vectors, private_vectors = torch.split(context_vectors, self.hidden // 2, -1)
		diff_loss = torch.bmm(private_vectors, torch.transpose(public_vectors, 2, 1))
		diff_loss = torch.sum(diff_loss ** 2) / context_vectors.size(0)

		return token_logits, next_logits, language_logits, diff_loss


class AdversarialPretrainingWrapper:
	"""
	adds pretraining tasks to entire multilingual model
	"""
	def __init__(self, multilingual_model, adversary_model, hidden, vocab_size):
		self.multilingual_model = multilingual_model
		self.adversary_model = adversary_model
		# add necessary prediction task
		self.pretraining_models = [AdversarialLanguageWrapper(model, adversary_model, hidden, vocab_size) for model in self.multilingual_model.language_models]

	def __getitem__(self, index):
		if type(index) is str:
			return self.pretraining_models[self.multilingual_model.ltoi[index]]
		else:
			return self.pretraining_models[index]
	
	def __len__(self):
		return len(self.multilingual_model)

	def adversary_forward(self, *args, **kwargs):
		context_vectors = self.multilingual_model.public(*args, **kwargs)
		return self.adversary_model(context_vectors)

