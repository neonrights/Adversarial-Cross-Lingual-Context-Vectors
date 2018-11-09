import json
import random


class DiscriminatorSequenceGenerator:
	"""
	Generates data file to be used to train a language discriminator.
	Data file is expected to be read by discriminator dataset
	"""
	def __init__(self, language_corpora, max_seq_len=512):
		"""
		language_corpora: a dictionary with language name as key and a list of corpus
			readers corresponding to a corpus in that language
		"""
		self.languages = set(language_corpora.keys())
		self.max_seq_len = max_seq_len
		self.language_corpora = language_corpora

	def write_samples(self, n_samples, out_path):
		with open(out_path, 'w+') as f_out:
			for i in n_samples:
				f_out.writeline(json.dumps(self.random_sample()))

	def random_sample(self):
		language = random.choice(self.languages)
		corpus = random.choice(self.language_corpora[language])
		document = random.choice(corpus)

		start = random.randint(len(sentences) - 5)
		end = start + 1

		seq_len = len(document['sentences'][start])
		while end < len(document['sentences']) and seq_len < self.max_seq_len:
			seq_len += len(document['sentences'][end])
			end += 1

		return {"language": language, "sentences": document['sentences'][start:end]}

