import pickle
import random


class LanguageSequenceGenerator:
	"""
	Generates data file to train or test a model for a specific language.
	Data file is expected to be read by languagedataset.
	"""
	def __init__(self, corpus, max_seq_len=512):
		"""
		corpus: an instance of corpusreader or a child class
		"""
		self.max_seq_len = max_seq_len
		self.corpus = corpus
	
	def random_samples(self, n_samples, out_path=None):
		if out_path is None:
			for i in range(n_samples):
				document = random.choice(self.corpus)
				sample = self.sample_sentences(document)

				if sample is None:
					print("Failed to write {} samples, got to {}".format(n_samples, i+1))
					break

				yield pickle.dumps(sample) + '\n'
		else:
			# write directly to file otherwise
			with open(out_path, 'wb+') as f_out:
				for i in range(n_samples):
					document = random.choice(self.corpus)
					sample = self.sample_sentences(document)

					if sample is None:
						print("Failed to write {} samples, got to {}".format(n_samples, i+1))
						break

					f_out.write(pickle.dumps(sample) + '\n')

			print("\tgenerated {} samples".format(n_samples))

	def sequential_samples(self, out_path=None):
		if out_path is None:
			for i, document in enumerate(self.corpus):
				sample = self.sample_sentences(document)

				if sample is None:
					print("Failed to sample from all documents, got to {}".format(i+1))
					break

				yield pickle.dumps(sample) + '\n'
		else:
			# write directly to file otherwise
			with open(out_path, 'wb+') as f_out:
				for i, document in enumerate(self.corpus):
					sample = self.sample_sentences(document)

					if sample is None:
						print("Failed to sample from all documents, got to {}".format(i+1))
						break

					f_out.write(pickle.dumps(sample) + '\n')

		print("\tgenerated {} samples".format(i))

	def sample_sentences(self, document):
		start = random.randint(0, len(document) - 5)
		end = start + 1

		seq_len = len(document['sentences'][start])
		while end < len(document['sentences']) and seq_len < self.max_seq_len:
			seq_len += len(document['sentences'][end])
			end += 1

		return dict((key, value[start:end]) if type(value) is list else (key, value)
				for key, value in document.items())

