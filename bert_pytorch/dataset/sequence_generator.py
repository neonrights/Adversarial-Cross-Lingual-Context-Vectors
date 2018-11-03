import random

# for monolingual, create json on each line
# json contains sequence of in language tokens
# for bilingual sequences, add bilingual sequence and token alignment
# add bilingual sentence continuation as well
class MonolingualSequenceGenerator:
	def __init__(self, mono_corpus, sent_corpus, text_corpus, max_seq_len, prioritize_corpora=True):
		self.max_seq_len = max_seq_len
		self.prioritize_corpora = prioritize_corpora
		self.corpora = [text_corpus, sent_corpus, mono_corpus]
	
	def generate_samples(self, n_samples, out_path):
		with open(out_path, 'w+') as f_out:
			for i in n_samples:
				if prioritize_corpora:
					sample = self.priority_sample()
				else:
					sample = self.random_sample()

				if sample is None:
					print("Failed to write {} samples, got to {}".format(n_samples, i))
					break

				f_out.writeline(json.dumps(sample))
					

	def random_sample(self):
		corpus = random.choice(self.corpora)
		sentences = corpus.
		# sample multiple sentences
		# write alt words and alignment if available
		# write alt sentence and alignment if available
		# write to dict
		sample = {
			"sentences": [],
			"alt_text": [],
			"text_alignment": [],
			"alt_sentences": [],
			"sentence_alignment": [],
			"alt_language": ""
		}
		return sample

	def priority_sample(self):
		# start with text aligned in order
		# if out of text aligned then sentence aligned
		# if out of sentence aligned then monolingual
		# write alt words and alignment if available
		# write alt sentence and alignment if available
		# write to dict


# for discriminator, have tsv file
# each line contains language of origin and sequence of language tokens split by tab
class DiscriminatorSequenceGenerator:
	def __init__(self, language_corpora, max_seq_len=512):
		self.languages = set(language_corpora.keys())
		self.max_seq_len = max_seq_len

		# open file handle for each corpus

	def write_samples(self, n_samples, out_path):
		with open(out_path, 'w+') as f_out:
			for i in n_samples:
				language, sentences = self.random_sample()
				f_out.write("{}\t{}\n".format(language, sentences))

	def random_sample(self):
		# select random language
		# select random corpus from language
		# sample multiple sentences of at most max_seq_len
		return language, sentences

