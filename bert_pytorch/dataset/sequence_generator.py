import json
import random

# for monolingual, create json on each line
# json contains sequence of in language tokens
# for bilingual sequences, add bilingual sequence and token alignment
# add bilingual sentence continuation as well
class MonolingualSequenceGenerator:
	def __init__(self, corpus, max_seq_len=512):
		self.max_seq_len = max_seq_len
		self.corpus = corpus
	
	def random_samples(self, n_samples, out_path):
		with open(out_path, 'w+') as f_out:
			for i in range(n_samples):
				document = random.choice(self.corpus)
				sample = self.sample_sentences(document)

				if sample is None:
					print("Failed to write {} samples, got to {}".format(n_samples, i+1))
					break

				f_out.write(json.dumps(sample) + '\n')

		print("generated {} samples to {}".format(n_samples, out_path))

	def sequential_samples(self, out_path):
		with open(out_path, 'w+') as f_out:
			for i, document in enumerate(self.corpus):
				sample = self.sample_sentences(document)

				if sample is None:
					print("Failed to sample from all documents, got to {}".format(i+1))
					break

				f_out.write(json.dumps(sample) + '\n')

		print("generated {} samples to {}".format(i, out_path))

	def sample_sentences(self, document):
		start = random.randint(0, len(document) - 5)
		end = start + 1

		seq_len = len(document[start]['text'])
		while end < len(document) and seq_len < self.max_seq_len:
			seq_len += len(document[end]['text'])
			end += 1
		
		sample = {"sentences": [sentence['text'] for sentence in document[start:end]]}
		if "alt_text" in document[start]:
			sample["alt_sentences"] = [sentence["alt_text"] for sentence in document[start:end]]
			sample["alt_language"] = document[start]["alt_language"]
			if "text_alignment" in document[start]:
				sample["text_alignment"] = [sentence["text_alignment"] for sentence in document[start:end]]

		return sample


# for discriminator, have tsv file
# each line contains language of origin and sequence of language tokens split by tab
class DiscriminatorSequenceGenerator:
	def __init__(self, language_corpora, max_seq_len=512):
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

		seq_len = len(document[start]['text'])
		sentences = [document[start]['text']]
		while end < len(sentences) and seq_len < self.max_seq_len:
			sentences.append(document[end]['text'])
			seq_len += len(document[end]['text'])
			end += 1

		return {"language": language, "sentences": sentences}


if __name__ == '__main__':
	print("Running smoke tests")
