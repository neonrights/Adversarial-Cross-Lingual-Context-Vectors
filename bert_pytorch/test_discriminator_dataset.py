from torch.utils.data import DataLoader

from dataset import DiscriminatorDataset, JSONVocab

print("Running unit tests for DiscriminatorDataset")

vocab = JSONVocab('test_discriminator.txt')
D_data = DiscriminatorDataset('test_discriminator.txt', vocab, {'en': 0, 'cz': 1}, seq_len=256)
D_loader = DataLoader(D_data, batch_size=32, shuffle=True)
print("passed on memory initialization tests")

for input_ids, labels in D_loader:
	assert input_ids.shape == (32, 256), "got {} for input shape instead of (32, 256)".format(inputs_ids.shape)
	assert labels.shape == (32,), "got {} for label shape instead of (32,)".format(labels.shape)
	break

print("passed on memory batch tests")

D_data = DiscriminatorDataset('test_discriminator.txt', vocab, {'en': 0, 'cz': 1}, seq_len=256, on_memory=False)
D_loader = DataLoader(D_data, batch_size=32, shuffle=True)
print("passed off memory initialization tests")

for input_ids, labels in D_loader:
	assert input_ids.shape == (32, 256), "got {} for input shape instead of (32, 256)".format(inputs_ids.shape)
	assert labels.shape == (32,), "got {} for label shape instead of (32,)".format(labels.shape)
	break

print("passed off memory batch tests")

D_data = DiscriminatorDataset('test_discriminator.txt', vocab, {'en': 0, 'cz': 1}, seq_len=256)
D_loader = DataLoader(D_data, batch_size=32, shuffle=True, num_workers=4)
print("passed multiple worker initialization tests")

for input_ids, labels in D_loader:
	assert input_ids.shape == (32, 256), "got {} for input shape instead of (32, 256)".format(inputs_ids.shape)
	assert labels.shape == (32,), "got {} for label shape instead of (32,)".format(labels.shape)
	break

print("passed multiple worker batch tests")