from torch.utils.data import DataLoader

from dataset import DiscriminatorDataset, BertTokenizer
import tqdm

print("Running unit tests for DiscriminatorDataset")

batch_size = 8
max_seq_len = 256
ltoi = {'ar': 0, 'bg': 1, 'de': 2, 'en': 3}

tokenizer = BertTokenizer('data/bert-base-multilingual-cased-vocab.txt')
D_data = DiscriminatorDataset("sample.disc.raw.txt", tokenizer, ltoi, max_seq_len)
D_loader = DataLoader(D_data, batch_size=batch_size, shuffle=True)
print("passed on memory initialization tests")

for i, batch in enumerate(D_loader):
	assert batch["input_ids"].shape == (batch_size, max_seq_len),\
			"got {} for input shape instead of {}".format(inputs_ids.shape, (batch_size, max_seq_len))
	assert batch["language_label"].shape == (batch_size,),\
			"got {} for label shape instead of {}".format(labels.shape, (batch_size,))
	if i > 1000:
		break

print("passed on memory batch tests")

D_data = DiscriminatorDataset('sample.disc.raw.txt', tokenizer, ltoi, max_seq_len, on_memory=False)
D_loader = DataLoader(D_data, batch_size=batch_size, shuffle=True)
print("passed off memory initialization tests")

for i, batch in enumerate(D_loader):
	assert batch["input_ids"].shape == (batch_size, max_seq_len),\
			"got {} for input shape instead of {}".format(inputs_ids.shape, (batch_size, max_seq_len))
	assert batch["language_label"].shape == (batch_size,),\
			"got {} for label shape instead of {}".format(labels.shape, (batch_size,))
	if i > 1000:
		break

print("passed off memory batch tests")

D_data = DiscriminatorDataset('sample.disc.raw.txt', tokenizer, ltoi, max_seq_len)
D_loader = DataLoader(D_data, batch_size=batch_size, shuffle=True, num_workers=4)
print("passed multiple worker initialization tests")

for i, batch in enumerate(D_loader):
	assert batch["input_ids"].shape == (batch_size, max_seq_len),\
			"got {} for input shape instead of {}".format(inputs_ids.shape, (batch_size, max_seq_len))
	assert batch["language_label"].shape == (batch_size,),\
			"got {} for label shape instead of {}".format(labels.shape, (batch_size,))
	if i > 1000:
		break

print("passed multiple worker batch tests")