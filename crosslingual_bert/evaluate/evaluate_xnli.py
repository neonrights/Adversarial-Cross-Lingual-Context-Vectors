import torch
from collections import Counter

import tqdm


def batch_BLEU(predicted, truth, predicted_mask=None, truth_mask=None, weights=(0.25, 0.25, 0.25, 0.25)):
	# calculates BLEU score given a bunch of ids
	scores = torch.zeros(predicted.size(0))
	for n, weight in enumerate(weights):
		n += 1
		# calculate BLEU score for each sample in batch at once


class EvaluateXNLI:
	def __init__(self, model, languages, target_language, vocab, with_cuda=True):
		cuda_condition = torch.cuda.is_available() and with_cuda
		self.device = torch.device("cuda:0" if cuda_condition else "cpu")

		# complete translation model with prediction
		self.model = model
		self.vocab = vocab

		# load train and test data
		self.xnli_data = xnli_data

		self.languages = languages
		self.target_language = target_language

	def evaluate(self, data_loader, save_file=None):
		# optimization and prediction
		data_iter = tqdm.tqdm(enumerate(data_loader), desc="XNLI BLEU", total=len(data_loader))
		total_scores, total_counts = Counter(), Counter()

		for i, batch in data_iter:
			batch = {key: value.to(self.device) for key, value in batch.items()}
			# forward iteration with mixture of experts	

			ground_truth = batch[self.target_language]
			for language in self.languages:
				token_ids, token_masks = batch[language], batch[language + "_mask"]

				translations = torch.empty_like(ground_truth)
				translations[:,0] = self.vocab.sos_index

				translation_mask = torch.ones_like(ground_truth)
				translations_mask[:,0] = 0
				seen_eos = torch.zeros(ground_truth.size(0))

				# generate translated token for each index in string
				for i in range(1, ground_truth.size(1)):
					token_logits = self.model[language](token_ids, translations, token_masks, translation_mask)
					token_predictions = logits.argmax(dim=-1)
					translations[:,i] = token_predictions
					seen_eos = seen_eos & token_predictions == (self.vocab.eos_index)
					translation_mask[:,i] = seen_eos

				# calculate BLEU scores
				translation_mask[:,0] = 1
				scores = batch_BLEU(translations, ground_truth, translation_mask, truth_mask)

				BLEU_scores[language] += scores.sum().item()
				sample_counts[language] += scores.nelements()

		BLEU_scores = {language: (total_scores[language] / total_counts[language]) for language in self.languages}
		for language in self.languages:
			print("{}: {}".format(language, BLEU_scores[language]))

		if save_file is not None:
			with open(save_file, 'w+') as f_out:
				for language in self.languages:
					f_out.write("{}\t{}\n".format(language), BLEU_scores[language])

			print("Written results to {}".format(save_file))

