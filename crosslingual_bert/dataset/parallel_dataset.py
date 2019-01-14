import random
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class ParallelDataset(Dataset):
	def __init__(self, parallel_tsv, tokenizer, seq_len, languages=None):
		self.tokenizer = tokenizer
		self.seq_len = seq_len
		self.parallel_df = pd.read_csv(parallel_tsv, sep='\t', usecols=languages)	

	def _sentence_to_ids(self, sentence):
		tokens = self.tokenizer.tokenize(sentence)
		tokens = ['[CLS]'] + tokens + ['[SEP]']
		ids = self.tokenizer.convert_tokens_to_ids(tokens[:self.seq_len])
		padding = [0 for _ in range(self.seq_len - len(ids))]
		return torch.tensor(ids + padding)

	def __getitem__(self, index):
		sample = self.parallel_df.iloc[index]
		return {language: self._sentence_to_ids(text) for language, text in sample.items()}

	def __len__(self):
		return len(self.parallel_df)


class ParallelTrainDataset(ParallelDataset):
	def __init__(self, parallel_tsv, tokenizer, seq_len, input_language, target_language):
		super().__init__(parallel_tsv, tokenizer, seq_len, languages=(input_language, target_language))
		self.input_language = input_language
		self.target_language = target_language

	def _sentence_to_ids_and_label(self, sentence):
		tokens = self.tokenizer.tokenize(sentence)
		tokens = ['<S>'] + tokens + ['<T>']
		split = random.randrange(1,min(len(tokens), self.seq_len))
		ids = self.tokenizer.convert_tokens_to_ids(tokens[:split])
		label = ids[-1]
		ids = ids[:-1]
		padding = [0 for _ in range(self.seq_len - len(ids))]
		return torch.tensor(ids + padding), torch.tensor(label)

	def __getitem__(self, index):
		sample = self.parallel_df.iloc[index]
		input_ids = self._sentence_to_ids(sample[self.input_language])
		target_ids, label = self._sentence_to_ids_and_label(sample[self.target_language])
		input_mask = input_ids == 0
		target_mask = target_ids == 0
		return {
			'input_ids': input_ids,
			'target_ids': target_ids,
			'input_mask': input_mask,
			'target_mask': target_mask,
			'labels': label
		}

	def __len__(self):
		return len(self.parallel_df)

