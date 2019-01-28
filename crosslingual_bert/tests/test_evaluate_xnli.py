import torch
import unittest
import unittest.mock as mock

from xnltk.translate import bleu_score

from dataset import BertTokenizer, ParallelDataset
from evaluate import EvaluateXNLI, batch_BLEU


class TestBatchBLEU(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokenizer = BertTokenizer("data/bert-base-multilingual-cased-vocab")

	def test_batch_BLEU(self):
		test_candidates = ["The villa was on fire today", "I severely dislike pickles", "The quick red fox jumped over the lazy brown dog"]
		test_references = ["A house was on fire today", "I hate vinegared cucumbers", "A quick red fox hurdled over a sleeping brown dog"]

		seq_len = 100

		# generate samples using known implementation
		batch_candidates = torch.zeros(len(test_candidates), seq_len, dtype=torch.long)
		batch_references = torch.zeros_like(batch_candidates)
		candidate_mask = torch.zeros_like(batch_candidates)
		reference_mask = torch.zeros_like(batch_candidates)
		reference_scores = torch.zeros(len(test_candidates))

		for i, candidate, reference in zip(range(len(test_candidates)), test_candidates, test_references):
			candidate_ids = self.tokenizer.tokenize_and_convert_to_ids(candidate)
			reference_ids = self.tokenizer.tokenize_and_convert_to_ids(reference)
			
			batch_candidates[i, :len(candidate_ids)] = candidate_ids
			batch_references[i, :len(reference_ids)] = reference_ids
			candidate_mask[i, :len(candidate_ids)] = 1
			reference_mask[i, :len(reference_ids)] = 1
			reference_scores[i] = bleu_score([reference_ids], candidate_ids)

		# compare scores between known and own implementation
		batch_scores = batch_BLEU(batch_candidates, batch_references, candidate_mask, reference_mask)
		for reference_score, batch_score in zip(reference_scores, batch_scores):
			self.assertAlmostEqual(reference_score, batch_score)


class TestEvaluateXNLI(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.tokenizer = BertTokenizer("data/bert-base-multilingual-cased-vocab.txt")
		self.model = mock.MagicMock()
		self.model.__call__.return_value = [
			# return set of predicted ids
		]

			# implement methods needed for mock
		self.languages = ['en']
		self.evaluator = EvaluateXNLI(self.model, self.tokenizer, self.languages)

	def test_score_generation(self):
		def test_sentences():
			# yield test values
			yield None

		dataloader = mock.MagicMock()
		dataloader.__iter__.return_value = test_sentences


if __name__ == '__main__':
	unittest.main()