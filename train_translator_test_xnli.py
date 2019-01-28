import torch
from torch.utils.data import DataLoader

from crosslingual_bert import ParallelDataset, ParallelTrainDataset, BertTokenizer
from crosslingual_bert import TranslatorTrainer
from crosslingual_bert import MultilingualTranslator, BertConfig
from crosslingual_bert import EvaluateXNLI

# initialize hyperparameters
seq_len = 180
languages = ['ar', 'bg', 'de']
target_language = 'en'
tokenizer = BertTokenizer("./example_data/bert-base-multilingual-cased-vocab.txt")
config = BertConfig(len(tokenizer.vocab),
		hidden_size=192,
		num_hidden_layers=3,
		num_attention_heads=12,
		intermediate_size=394,
		hidden_act='gelu',
		max_position_embeddings=256)

# load datasets
ON_MEMORY = False
ar_en_raw = ParallelTrainDataset('./example_data/sample.ar-en.tsv', tokenizer, seq_len, 'ar', 'en')
bg_en_raw = ParallelTrainDataset('./example_data/sample.bg-en.tsv', tokenizer, seq_len, 'bg', 'en')
de_en_raw = ParallelTrainDataset('./example_data/sample.de-en.tsv', tokenizer, seq_len, 'de', 'en')

ar_en_data = DataLoader(ar_en_raw, batch_size=16, shuffle=True, num_workers=4)
bg_en_data = DataLoader(ar_en_raw, batch_size=16, shuffle=True, num_workers=4)
de_en_data = DataLoader(ar_en_raw, batch_size=16, shuffle=True, num_workers=4)

train_data = {
	'ar': ar_en_data,
	'bg': bg_en_data,
	'de': de_en_data
}

# load multilingual model
multi_model = torch.load('./example_data/sample.model')
multi_model = multi_model.multilingual_model
translator_model = MultilingualTranslator(multi_model, 'en', config)

# train model
trainer = TranslatorTrainer(translator_model, languages, target_language,
		train_data, train_data)

best_epoch = 0
best_loss = 1e9
for epoch in range(100):
	epoch += 1
	trainer.train(epoch)
	test_loss = trainer.test(epoch)
	if test_loss < best_loss:
		best_epoch = epoch
		best_loss = test_loss
		trainer.save(epoch, 'translator', 'best.model')
	if epoch % 10 == 0:
		trainer.save(epoch, 'translator')

# evaluate model
xnli_raw = ParallelDataset('./example_data/xnli.15way.orig.tsv',
		tokenizer, seq_len, languages, languages + [target_language])
xnli_data = DataLoader(xnli_raw, batch_size=16, num_workers=4)
evaluator = EvaluateXNLI(translator_model, tokenizer, languages, target_language)
evaluator.evalutate(xnli_data, './translator/results.txt')