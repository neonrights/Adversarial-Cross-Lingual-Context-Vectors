import unittest
import tqdm
from torch.utils.data import DataLoader

from dataset import DiscriminatorDataset, BertTokenizer

print("Running unit tests for DiscriminatorDataset")

class TestDiscriminatorDataset(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.batch_size = 8
		self.max_seq_len = 256
		self.ltoi = {'ar': 0, 'bg': 1, 'de': 2, 'en': 3}
		self.num_workers = 8
		self.tokenizer = BertTokenizer('data/bert-base-multilingual-cased-vocab.txt')

	def test_on_memory(self):
		D_data = DiscriminatorDataset("sample.disc.raw.txt", self.tokenizer, self.ltoi, self.max_seq_len)
		D_loader = DataLoader(D_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

		for i, batch in tqdm.tqdm(enumerate(D_loader), total=len(D_loader), desc="on memory test"):
			self.assertTrue(batch["input_ids"].shape == (self.batch_size, self.max_seq_len),
					"got {} for input shape instead of {}".format(batch["input_ids"].shape, (self.batch_size, self.max_seq_len)))
			self.assertTrue(batch["language_label"].shape == (self.batch_size,),
					"got {} for label shape instead of {}".format(batch["language_label"].shape, (self.batch_size,)))

	def test_off_memory(self):
		D_data = DiscriminatorDataset('sample.disc.raw.txt', self.tokenizer, self.ltoi, self.max_seq_len, on_memory=False)
		D_loader = DataLoader(D_data, batch_size=self.batch_size, drop_last=True)

		for i, batch in tqdm.tqdm(enumerate(D_loader), total=len(D_loader), desc="off memory test"):
			self.assertTrue(batch["input_ids"].shape == (self.batch_size, self.max_seq_len),
					"got {} for input shape instead of {}".format(batch["input_ids"].shape, (self.batch_size, self.max_seq_len)))
			self.assertTrue(batch["language_label"].shape == (self.batch_size,),
					"got {} for label shape instead of {}".format(batch["language_label"].shape, (self.batch_size,)))

	def test_on_memory_workers(self):
		D_data = DiscriminatorDataset('sample.disc.raw.txt', self.tokenizer, self.ltoi, self.max_seq_len)
		D_loader = DataLoader(D_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

		for i, batch in tqdm.tqdm(enumerate(D_loader), total=len(D_loader), desc="on memory w/ workers"):
			self.assertTrue(batch["input_ids"].shape == (self.batch_size, self.max_seq_len),
					"got {} for input shape instead of {}".format(batch["input_ids"].shape, (self.batch_size, self.max_seq_len)))
			self.assertTrue(batch["language_label"].shape == (self.batch_size,),
					"got {} for label shape instead of {}".format(batch["language_label"].shape, (self.batch_size,)))

	def test_off_memory_workers(self):
		D_data = DiscriminatorDataset('sample.disc.raw.txt', self.tokenizer, self.ltoi, self.max_seq_len, on_memory=False)
		D_loader = DataLoader(D_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

		for i, batch in tqdm.tqdm(enumerate(D_loader), total=len(D_loader), desc="off memory w/ workers"):
			self.assertTrue(batch["input_ids"].shape == (self.batch_size, self.max_seq_len),
					"got {} for input shape instead of {}".format(batch["input_ids"].shape, (self.batch_size, self.max_seq_len)))
			self.assertTrue(batch["language_label"].shape == (self.batch_size,),
					"got {} for label shape instead of {}".format(batch["language_label"].shape, (self.batch_size,)))


if __name__ == '__main__':
	unittest.main()