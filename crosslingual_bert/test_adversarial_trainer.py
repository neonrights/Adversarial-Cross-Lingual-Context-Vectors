from torch.utils.data import DataLoader

from model import BertConfig, BertModel, MultilingualBERT
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
hidden = 256
config = BertConfig(vocab_size=len(vocab),
		hidden_size=hidden//2,
		num_hidden_layers=6,
		num_attention_heads=8,
		intermediate_size=hidden,
		max_position_embeddings=256)

model = MultilingualBERT(language_ids, BertModel, config)
adversary = DummyAdversary(hidden//2, len(language_ids))
trainer = AdversarialPretrainer(model, adversary, len(vocab), hidden, language_ids, train_data, D_dataset, train_data, 5, beta=0.1, gamma=1e-9)

for epoch in range(1000):
	trainer.train(epoch)

	if (epoch+1) % 10 == 0:
		trainer.save(epoch)
