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
	def __init__(self, model, languages, target_language, with_cuda=True):

		cuda_condition = torch.cuda.is_available() and with_cuda
		self.device = torch.device("cuda:0" if cuda_condition else "cpu")

		# complete translation model with prediction
		self.model = model
		# load train and test data
		self.xnli_data = xnli_data

		self.languages = languages
		self.target_language = target_language

	def evaluate(self, data_loader, save_file=None):
		# optimization and prediction
		data_iter = tqdm.tqdm(enumerate(data_loader),
							  desc="XNLI BLEU",
							  total=len(data_loader))

		total_scores, total_counts = Counter(), Counter()
		for i, batch in data_iter:
			batch = {key: value.to(self.device) for key, value in batch.items()}
			# forward iteration with mixture of experts	

			ground_truth = batch[self.target_language]
			for language in self.languages:
				# calculate until max seq len or eos token is reached
				# calculate masks for samples
				# calculate BLEU scores
				scores = batch_BLEU(predicted, truth, predicted_mask, truth_mask)
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

