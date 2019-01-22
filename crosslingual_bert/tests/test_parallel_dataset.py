from torch.utils.data import DataLoader
from dataset import ParallelDataset, BertTokenizer


print("Running unittests for XNLI dataset...")
batch_size = 12
seq_len = 128

tokenizer = BertTokenizer("data/bert-base-multilingual-uncased-vocab.txt")
dataset = ParallelDataset("data/xnli.15way.orig.tsv", tokenizer, seq_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("passed initialization and dataloader tests")

for i, batch in enumerate(data_loader):
	assert type(batch) is dict
	assert len(batch.keys()) == 15
	assert batch['en'].shape == (batch_size, seq_len)
	if i > 10:
		break

print("passed batch sampling tests")

languages = ('vi', 'en')
tokenizer = BertTokenizer("data/bert-base-multilingual-uncased-vocab.txt")
dataset = ParallelDataset("data/xnli.15way.orig.tsv", tokenizer, seq_len, languages=languages)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("passed initialization of subset of columns")

for i, batch in enumerate(data_loader):
	assert type(batch) is dict
	assert batch.keys() == set(languages)
	assert batch['en'].shape == (batch_size, seq_len)
	if i > 10:
		break

print("passed batch subset sampling")