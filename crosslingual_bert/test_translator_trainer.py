import torch
from torch.utils.data import DataLoader

from dataset import *
from model import MultilingualBert, MultilingualTranslator,\
		BertConfig, BertModel, TranslatorModel
from trainer import TranslatorTrainer

print("Running unit tests for TranslatorTrainer...")

language_ids = {'en': 0, 'cz': 1, 'fr': 2, 'ch': 3, 'jp': 4}
vocab_size = 1000
batch_size = 16
seq_len = 64

config = BertConfig(vocab_size,
		hidden_size=300,
		num_hidden_layers=3,
		intermediate_size=450)
embedder = MultilingualBert(language_ids, BertModel, config)

config.hidden_size *= 2
config.intermediate_size *= 2
translator = TranslatorModel(config)
multi_translator = MultilingualTranslator(embedder, translator, 'en')

# set up language dataset
sample_data = dict()

trainer = TranslatorTrainer(multi_translator, languages, sample_data, sample_data)

for epoch in range(5):
	trainer.train(epoch)
	trainer.test(epoch)

