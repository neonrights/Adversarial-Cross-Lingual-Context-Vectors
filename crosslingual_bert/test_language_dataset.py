import unittest
import tqdm
from torch.utils.data import DataLoader

from dataset import LanguageDataset, JSONVocab, BertTokenizer


class TestLanguageDataset(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.batch_size = 16
		self.max_seq_len = 256
		self.num_workers = 8
		self.tokenizer = BertTokenizer('data/bert-base-multilingual-cased-vocab.txt')

	def test_on_memory(self):
		language_data = LanguageDataset('en', 'all.sample.en.txt', self.tokenizer, max_seq_len=self.max_seq_len)
		language_loader = DataLoader(language_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

		for i, batch in tqdm.tqdm(enumerate(language_loader), total=len(language_loader), desc="on memory test"):
			self.assertTrue(type(batch) is dict, "expected dict for batch object, got {}".format(type(batch)))
			self.assertTrue(batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next", "mask"}, 
					"did not get expected keys in batch dict")
			for key, value in batch.items():
				if len(value.shape) == 1:
					self.assertTrue(value.shape == (self.batch_size,), "expected {} but got {}".format((self.batch_size,), value.shape))
				else:
					self.assertTrue(value.shape[:2] == (self.batch_size, self.max_seq_len), "expected {} but got {}".format((self.batch_size, self.max_seq_len), value.shape[:2]))

	def test_off_memory(self):
		language_data = LanguageDataset('en', 'all.sample.en.txt', self.tokenizer, max_seq_len=self.max_seq_len, on_memory=False)
		language_loader = DataLoader(language_data, batch_size=self.batch_size, drop_last=True)

		for i, batch in tqdm.tqdm(enumerate(language_loader), total=len(language_loader), desc="off memory test"):
			self.assertTrue(type(batch) is dict, "expected dict for batch object, got {}".format(type(batch)))
			self.assertTrue(batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next", "mask"}, 
					"did not get expected keys in batch dict")
			for key, value in batch.items():
				if len(value.shape) == 1:
					self.assertTrue(value.shape == (self.batch_size,), "expected {} but got {}".format((self.batch_size,), value.shape))
				else:
					self.assertTrue(value.shape[:2] == (self.batch_size, self.max_seq_len), "expected {} but got {}".format((self.batch_size, self.max_seq_len), value.shape[:2]))

	def test_on_memory_workers(self):
		language_data = LanguageDataset('en', 'all.sample.en.txt', self.tokenizer, max_seq_len=self.max_seq_len)
		language_loader = DataLoader(language_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

		for i, batch in tqdm.tqdm(enumerate(language_loader), total=len(language_loader), desc="on memory w/ workers"):
			self.assertTrue(type(batch) is dict, "expected dict for batch object, got {}".format(type(batch)))
			self.assertTrue(batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next", "mask"}, 
					"did not get expected keys in batch dict")
			for key, value in batch.items():
				if len(value.shape) == 1:
					self.assertTrue(value.shape == (self.batch_size,), "expected {} but got {}".format((self.batch_size,), value.shape))
				else:
					self.assertTrue(value.shape[:2] == (self.batch_size, self.max_seq_len), "expected {} but got {}".format((self.batch_size, self.max_seq_len), value.shape[:2]))

	def test_off_memory_workers(self):
		language_data = LanguageDataset('en', 'all.sample.en.txt', self.tokenizer, max_seq_len=self.max_seq_len, on_memory=False)
		language_loader = DataLoader(language_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

		for i, batch in tqdm.tqdm(enumerate(language_loader), total=len(language_loader), desc="off memory w/ workers"):
			self.assertTrue(type(batch) is dict, "expected dict for batch object, got {}".format(type(batch)))
			self.assertTrue(batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next", "mask"}, 
					"did not get expected keys in batch dict")
			for key, value in batch.items():
				if len(value.shape) == 1:
					self.assertTrue(value.shape == (self.batch_size,), "expected {} but got {}".format((self.batch_size,), value.shape))
				else:
					self.assertTrue(value.shape[:2] == (self.batch_size, self.max_seq_len), "expected {} but got {}".format((self.batch_size, self.max_seq_len), value.shape[:2]))


if __name__ == '__main__':
	unittest.main()