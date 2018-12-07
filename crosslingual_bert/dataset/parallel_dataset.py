import torch
import pandas as pd

from torch.utils.data import Dataset


class ParallelDataset(Dataset):
	def __init__(self, parallel_tsv, tokenizer, seq_len, languages=None):
		self.tokenizer = tokenizer
		self.seq_len = seq_len
		self.parallel_df = pd.read_csv(parallel_tsv, sep='\t', usecols=languages)

	def _sentence_to_ids(self, sentence):
		tokens = self.tokenizer.tokenize(sentence)
		tokens = ['<S>'] + tokens + ['<T>']
		ids = self.tokenizer.convert_tokens_to_ids(tokens[:self.seq_len])
		padding = [0 for _ in range(self.seq_len - len(ids))]
		return torch.tensor(ids + padding)

	def __getitem__(self, index):
		sample = self.parallel_df.iloc[index]
		return {language: self._sentence_to_ids(text) for language, text in sample.items()}

	def __len__(self):
		return len(self.parallel_df)

