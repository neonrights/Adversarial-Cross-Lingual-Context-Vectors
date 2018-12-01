from torch.utils.data import DataLoader
from dataset import XNLIDataset, JSONVocab


print("Running unittests for XNLI dataset...")
batch_size = 12
seq_len = 128

vocab = JSONVocab.load_vocab("test_vocab.pkl")
dataset = XNLIDataset("data/xnli.15way.orig.tsv", vocab, seq_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("passed initialization and dataloader tests")

for i, batch in enumerate(data_loader):
	assert type(batch) is dict
	assert batch.keys() == dataset.languages
	assert batch['en'].shape == (batch_size, seq_len)
	if i > 10:
		break

print("passed batch sampling tests")