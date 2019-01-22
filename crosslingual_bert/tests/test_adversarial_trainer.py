from torch.utils.data import DataLoader

from model import BertConfig, MultilingualBert
from dataset import DiscriminatorDataset, LanguageDataset, JSONVocab
from trainer import AdversarialPretrainer


vocab = JSONVocab.load_vocab("test_vocab.pkl")
language_ids = {'en': 0, 'cz': 1}

en_dataset = LanguageDataset("test_english.txt", vocab, language='en', seq_len=128)
cz_dataset = LanguageDataset("test_czech.txt", vocab, language='cz', seq_len=128)
D_dataset = DiscriminatorDataset("test_discriminator.txt", vocab, language_ids, seq_len=128)

en_dataset = DataLoader(en_dataset, batch_size=32, shuffle=True)
cz_dataset = DataLoader(cz_dataset, batch_size=32, shuffle=True)
D_dataset = DataLoader(D_dataset, batch_size=32, shuffle=True)

train_data = {'en': en_dataset, 'cz': cz_dataset, 'adversary': D_dataset}

# initialize models
hidden = 256
config = BertConfig(vocab_size=len(vocab),
		hidden_size=hidden//2,
		num_hidden_layers=6,
		num_attention_heads=8,
		intermediate_size=hidden,
		max_position_embeddings=256)

model = MultilingualBert(language_ids, config)
trainer = AdversarialPretrainer(model,
		config,
		language_ids,
		train_data,
		train_data,
		2, beta=0.1, gamma=1e-4)

for epoch in range(5):
	trainer.train(epoch)
	trainer.save(epoch)
