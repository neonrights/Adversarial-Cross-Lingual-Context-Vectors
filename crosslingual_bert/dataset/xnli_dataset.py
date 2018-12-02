import torch
import pandas as pd

from torch.utils.data import Dataset


class XNLIDataset(Dataset):
	def __init__(self, xnli_tsv, vocab, seq_len, languages=None):
		self.vocab = vocab
		self.seq_len = seq_len
		self.xnli_df = pd.read_csv(xnli_tsv, sep='\t')
		if languages is not None:
			self.xnli_df = self.xnli_df[languages]
			self.languages = set(languages)
		else:
			self.languages = set(self.xnli_df.columns)

	def _sentence_to_ids(self, sentence):
		ids = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sentence.split()]
		ids = [self.vocab.sos_index] + ids + [self.vocab.eos_index]
		ids = ids[:self.seq_len]
		padding = [self.vocab.pad_index for _ in range(self.seq_len - len(ids))]
		return torch.tensor(ids + padding)

	def __getitem__(self, index):
		sample = self.xnli_df.iloc[index]
		return {language: self._sentence_to_ids(text) for language, text in sample.items()}

	def __len__(self):
		return len(self.xnli_df)

