from torch.utils.data import DataLoader

from model import BERT, MultilingualModel
from dataset import DiscriminatorDataset, LanguageDataset, JSONVocab
from trainer import AdversarialPretrainer
from dummy_adversary import DummyAdversary


vocab = JSONVocab.load_vocab("test_vocab.pkl")
language_ids = {'en': 0, 'cz': 1}

en_dataset = LanguageDataset("test_english.txt", vocab, language='en', seq_len=128)
cz_dataset = LanguageDataset("test_czech.txt", vocab, language='cz', seq_len=128)
D_dataset = DiscriminatorDataset("test_discriminator.txt", vocab, language_ids, seq_len=128)

en_dataset = DataLoader(en_dataset, batch_size=32, shuffle=True)
cz_dataset = DataLoader(cz_dataset, batch_size=32, shuffle=True)
D_dataset = DataLoader(D_dataset, batch_size=32, shuffle=True)

train_data = {'en': en_dataset, 'cz': cz_dataset}

# initialize models
hidden = 128
model = MultilingualModel(language_ids, BERT, vocab_size=len(vocab), hidden=hidden//2, n_layers=3, attn_heads=8, dropout=0.5)
adversary = DummyAdversary(hidden//2, len(language_ids))
trainer = AdversarialPretrainer(model, adversary, len(vocab), hidden, language_ids, train_data, D_dataset, train_data, 1)

for epoch in range(5):
	trainer.train(epoch)
	#trainer.save(epoch)
