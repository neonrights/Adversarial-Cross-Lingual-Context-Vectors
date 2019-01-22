from torch.utils.data import DataLoader
from dataset import ParallelTrainDataset, BertTokenizer


print("Running unittests for XNLI dataset...")
batch_size = 12
seq_len = 128

tokenizer = BertTokenizer("data/bert-base-multilingual-uncased-vocab.txt")
dataset = ParallelTrainDataset("data/xnli.15way.orig.tsv", tokenizer,
		target_language='vi', input_language='en', seq_len=seq_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("passed initialization and dataloader tests")

for i, batch in enumerate(data_loader):
	assert type(batch) is dict
	assert batch.keys() == {'input_ids', 'target_ids', 'input_mask',
			'target_mask', 'labels'}
	assert batch['input_ids'].shape == (batch_size, seq_len)
	assert batch['target_ids'].shape == (batch_size, seq_len)
	assert batch['input_mask'].shape == (batch_size, seq_len)
	assert batch['target_mask'].shape == (batch_size, seq_len)
	assert batch['labels'].shape == (batch_size,)
	if i > 10:
		break

print("passed batch sampling tests")