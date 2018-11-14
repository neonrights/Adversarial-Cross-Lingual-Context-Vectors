from torch.utils.data import DataLoader

from dataset import DiscriminatorDataset, JSONVocab

vocab = JSONVocab('test_discriminator.txt')
D_data = DiscriminatorDataset('test_discriminator.txt', vocab, {'en': 0, 'cz': 1}, seq_len=256)
D_loader = DataLoader(D_data, batch_size=32, shuffle=True, num_workers=2)

for batch in D_loader:
	# test batch
	break
