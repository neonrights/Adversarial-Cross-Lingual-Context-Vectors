import torch
from torch.utils.data import DataLoader

from dataset import ParallelDataset, ParallelTrainDataset, BertTokenizer
from model import MultilingualBert, MultilingualTranslator,\
		BertConfig, BertModel, TranslatorModel
from trainer import TranslatorTrainer


print("Running unit tests for TranslatorTrainer...")

language_ids = {'en': 0, 'vi': 1, 'fr': 2, 'zh': 3}
tokenizer = BertTokenizer("data/bert-base-multilingual-uncased-vocab.txt")
vocab_size = len(tokenizer.vocab)
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
sample_data = {}
for language in language_ids:
	if language != 'en':
		sample_data[language] = ParallelTrainDataset("data/xnli.15way.orig.tsv",
				tokenizer, seq_len, input_language=language, target_language='en')

sample_data = {key: DataLoader(value, batch_size=batch_size, shuffle=True)
		for key, value in sample_data.items()}
trainer = TranslatorTrainer(multi_translator, ['vi', 'fr', 'zh'], 'en',
		sample_data, sample_data)

for epoch in range(5):
	trainer.train(epoch)
	trainer.test(epoch)

