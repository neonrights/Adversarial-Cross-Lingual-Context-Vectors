from torch.utils.data import DataLoader
from dataset import LanguageDataset, JSONVocab, BertTokenizer
import tqdm

print("Running unit tests for LanguageDataset...")

batch_size = 8
max_seq_len = 256

tokenizer = BertTokenizer('data/bert-base-multilingual-cased-vocab.txt')
language_data = LanguageDataset('all.sample.en.txt', tokenizer, max_seq_len=max_seq_len)
language_loader = DataLoader(language_data, batch_size=batch_size, shuffle=True)
print("passed on memory initialization tests")

for i, batch in enumerate(language_loader):
	assert type(batch) is dict, "expected dict for batch object, got {}".format(type(batch))
	assert batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next", "mask"},\
			"did not get expected keys in batch dict"
	for key, value in batch.items():
		if len(value.shape) == 1:
			assert value.shape == (batch_size,), "expected {} but got {}".format((batch_size,), value.shape)
		else:
			assert value.shape[:2] == (batch_size, max_seq_len), "expected {} but got {}".format((batch_size, max_seq_len), value.shape[:2])

	if i > 1000:
		break

print("passed on memory batch tests")

language_data = LanguageDataset('all.sample.en.txt', tokenizer, max_seq_len=max_seq_len, on_memory=False)
language_loader = DataLoader(language_data, batch_size=batch_size, shuffle=True)
print("passed off memory initialization tests")

for i, batch in enumerate(language_loader):
	assert type(batch) is dict, "expected dict for batch object, got {}".format(type(batch))
	assert batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next", "mask"},\
			"did not get expected keys in batch dict"
	for key, value in batch.items():
		if len(value.shape) == 1:
			assert value.shape == (batch_size,), "expected {} but got {}".format((batch_size,), value.shape)
		else:
			assert value.shape[:2] == (batch_size, max_seq_len), "expected {} but got {}".format((batch_size, max_seq_len), value.shape[:2])

	if i > 1000:
		break

print("passed off memory batch tests")

language_data = LanguageDataset('all.sample.en.txt', tokenizer, max_seq_len=max_seq_len)
language_loader = DataLoader(language_data, batch_size=batch_size, shuffle=True, num_workers=4)
print("passed multiple worker initialization tests")

for i, batch in enumerate(language_loader):
	assert type(batch) is dict, "expected dict for batch object, got {}".format(type(batch))
	assert batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next", "mask"},\
			"did not get expected keys in batch dict"
	for key, value in batch.items():
		if len(value.shape) == 1:
			assert value.shape == (batch_size,), "expected {} but got {}".format((batch_size,), value.shape)
		else:
			assert value.shape[:2] == (batch_size, max_seq_len), "expected {} but got {}".format((batch_size, max_seq_len), value.shape[:2])

	if i > 1000:
		break

print("passed multiple worker batch tests")