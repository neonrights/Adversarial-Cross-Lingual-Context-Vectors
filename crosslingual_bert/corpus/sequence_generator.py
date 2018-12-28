import os
import sys
import argparse
import random
import tqdm
import multiprocessing

from readers import OpenSubtitlesReader


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
				while not sample or len(sample) == 1:
					document = random.choice(self.corpus)
					sample = self.sample_sentence(document)

				f_out.write('\t'.join(sample) + '\n')

	@staticmethod
	def worker_function(corpus, filename):
		return corpus.extract_sentences(filename)

	def sample_from_all(self, out_path, n_processes=None):
		"""generate a sample once from all documents in the corpus
		"""
		with open(out_path, 'w+') as f_out:		
			if not n_processes:
				skipped = 0
				for document in tqdm.tqdm(self.corpus, desc="writing samples"):
					sample = self.sample_sentence(document)

					if sample and len(sample) > 1:
						f_out.write('\t'.join(sample) + '\n')
					else:
						skipped += 1

			else:
				pool = multiprocessing.Pool(n_processes)
				document_iter = tqdm.tqdm(pool.imap(SequenceGenerator.worker_function,
						[(self.corpus, filename) for filename in self.corpus.files]))
				for document in document_iter:
					sample = self.sample_sentence(document)

					if sample and len(sample) > 1:
						f_out.write('\t'.join(sample) + '\n')
					else:
						skipped += 1

		print("skipped %d documents" % skipped)

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


if __name__ == '__main__':
	for filename in sys.argv[1:]:
		filepath, name = os.path.split(filename)
		out_name = name.split('.')[:2]
		out_name = '.'.join(out_name) + '.txt'
		out_path = os.path.join(filepath, out_name)

		reader = OpenSubtitlesReader(filename)
		generator = SequenceGenerator(reader)
		generator.sample_from_all(out_path)

