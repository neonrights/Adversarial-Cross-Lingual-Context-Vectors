from torch.utils.data import DataLoader
from dataset import LanguageDataset, JSONVocab

vocab = JSONVocab('test_discriminator.txt')
language_data = LanguageDataset('test_discriminator.txt', vocab, seq_len=256)
language_loader = DataLoader(language_data, batch_size=32, shuffle=True, num_workers=2)

for batch in language_loader:
	# test batch
	break
