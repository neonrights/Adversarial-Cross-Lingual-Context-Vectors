from torch.utils.data import DataLoader
from dataset import LanguageSwapDataset, JSONVocab

print("Running unit tests for LanguageSwapDataset...")

vocab = JSONVocab('test_language.txt')
language_data = LanguageSwapDataset('test_language.txt', vocab, language='en', seq_len=256)
language_loader = DataLoader(language_data, batch_size=32, shuffle=True)
print("passed on memory initialization tests")

for i, batch in enumerate(language_loader):
	assert type(batch) is dict, "expected dict for batch object, got {}".format(type(batch))
	assert batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next"},\
			"did not get expected keys in batch dict"
	if i > 10:
		break

print("passed on memory batch tests")

language_data = LanguageSwapDataset('test_language.txt', vocab, language='en', seq_len=256, on_memory=False)
language_loader = DataLoader(language_data, batch_size=32, shuffle=True)
print("passed off memory initialization tests")

for i, batch in enumerate(language_loader):
	assert type(batch) is dict, "expected dict for batch object, got {}".format(type(batch))
	assert batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next"},\
			"did not get expected keys in batch dict"
	if i > 10:
		break

print("passed off memory batch tests")

language_data = LanguageSwapDataset('test_language.txt', vocab, language='en', seq_len=256)
language_loader = DataLoader(language_data, batch_size=32, shuffle=True, num_workers=4)
print("passed multiple worker initialization tests")

for batch in language_loader:
	assert type(batch) is dict, "expected dict for batch object, got {}".format(type(batch))
	assert batch.keys() == {"input_ids", "token_labels", "segment_label", "is_next"},\
			"did not get expected keys in batch dict"
	if i > 10:
		break

print("passed multiple worker batch tests")