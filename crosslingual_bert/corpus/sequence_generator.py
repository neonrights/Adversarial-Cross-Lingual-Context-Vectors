

class SequenceGenerator:
	"""Generates tokenized sequences of sentences from a specific corpus given
	a corpus reader which returns a list of sentence strings and a tokenizer.
	Creates samples of contiguous sentences up to a set max number of tokens,
	then writes the samples to a file.  Each sentence is written on a line, with
	each sample separated by an empty line.
	"""
	def __init__(self, corpus_reader, tokenizer, max_seq_len=512):
		self.corpus = corpus_reader
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len

	def random_samples(self, n_samples, out_path):
		"""generate a sample from n randomly chosen documents in the corpus
		"""
		with open(out_path, 'w+') as f_out:
			for i in range(n_samples):
				for i in range(n_samples):
				document = random.choice(self.corpus)
				sample = self.sample_sentences(document)

				if sample is None:
					print("Failed to write {} samples, got to {}".format(n_samples, i+1))
					break

				f_out.write('\t'.join(sample) + '\n')

	def sample_from_all(self, out_path):
		"""generate a sample once from all documents in the corpus
		"""
		with open(out_path, 'w+') as f_out:
			for document in self.corpus:
				sample = self.sample_sentence(document)

				if sample is None:
					print("Failed to write samples")
					break

				f_out.write('\t'.join(sample) + '\n')


	def sample_sentence(self, document):
		start = random.randint(0, len(document) - 5)
		end = start + 1

		sample = self.tokenizer.tokenize(document[start])
		while end < len(document) and len(samples) < self.max_seq_len:
			sample += self.tokenizer.tokenize(document[end])
			end += 1

		return sample
