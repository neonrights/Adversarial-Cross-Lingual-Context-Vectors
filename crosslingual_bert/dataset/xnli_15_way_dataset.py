import torch
import pandas as pd

from torch.utils.data import Dataset


class XNLI15WayDataset(Dataset):
	def __init__(self, xnli_tsv, vocab, seq_len, languages=None):
		self.vocab = vocab
		self.seq_len = seq_len
		self.xnli_df = pd.read_csv(xnli_tsv, sep='\t')
		if language is not None:
			self.xnli_df = self.xnli_df[languages]

	def __getitem__(self, index):
		sample = self.dataframe.iloc[index]
		input_ids = {language: [self.vocab.stoi.get(token, self.vocab.unk_index) for token in text.split()]
				for language, text in sample.items()}
		input_ids = {language: [self.vocab.sos_index] + sentence + [self.vocab.eos_index]
				for language, sentence in input_ids.items()}
		input_ids = {language: sentence[:self.seq_len] for language, sentence in input_ids.items()}
		input_ids = {language: sentence + [self.vocab.pad_index for _ in range(self.seq_len - len(sentence))]
				for language, sentence in input_ids.items()}
		return {language: torch.tensor(ids) for language, ids in sample.items()}

	def __len__(self):
		return len(self.xnli_df)

