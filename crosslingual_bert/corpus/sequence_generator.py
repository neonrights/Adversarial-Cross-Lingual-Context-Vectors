import random
import tqdm
from .debugger import exception_debugger

class SequenceGenerator:
	"""Generates tokenized sequences of sentences from a specific corpus given
	a corpus reader which returns a list of sentence strings and a tokenizer.
	Creates samples of contiguous sentences up to a set max number of characters,
	then writes the samples to a file.  Each sentence is written on a line, with
	each sample separated by an empty line.
	"""
	def __init__(self, corpus_reader, max_seq_len=4096):
		self.corpus = corpus_reader
		self.max_seq_len = max_seq_len

	def random_samples(self, n_samples, out_path):
		"""generate a sample from n randomly chosen documents in the corpus
		"""
		with open(out_path, 'w+') as f_out:
			for i in tqdm.tqdm(range(n_samples), desc="writing samples"):
				sample = None
				while not sample:
					document = random.choice(self.corpus)
					sample = self.sample_sentence(document)

				f_out.write('\t'.join(sample) + '\n')

	def sample_from_all(self, out_path):
		"""generate a sample once from all documents in the corpus
		"""
		with open(out_path, 'w+') as f_out:
			for document in tqdm.tqdm(self.corpus, desc="writing samples"):
				sample = self.sample_sentence(document)

				if sample:
					f_out.write('\t'.join(sample) + '\n')

	def sample_sentence(self, document):
		if not document or len(document) <= 1:
			return []

		start = random.randrange(len(document)-1)
		end = start + 1

		char_count = len(document[start])
		while end < len(document) and char_count < self.max_seq_len:
			char_count += len(document[end])
			end += 1

		return document[start:end]
