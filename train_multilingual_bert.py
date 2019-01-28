import os
import torch

from multiprocessing import pool
from torch.utils.data import DataLoader

from crosslingual_bert.dataset import BertTokenizer, LanguageDataset, DiscriminatorDataset
from crosslingual_bert.model import MultilingualBert, MultilingualConfig
from crosslingual_bert.trainer import AdversarialPretrainer, AdversarialPretrainerConfig


# initialize hyperparameters
save_path = "large-1"
seq_len = 200 # XNLI max sequence length with wordpiece tokenization is 167
ltoi = {'ar': 0, 'bg': 1, 'de': 2, 'en': 3}
tokenizer = BertTokenizer("./example_data/bert-base-multilingual-cased-vocab.txt")

model_config = MultilingualConfig(
	languages=ltoi,
	vocab_size=len(tokenizer.vocab),
	hidden_size=384,
	intermediate_size=1536,
	max_position_embeddings=256,
	checkpoint_every=2
)

trainer_config = AdversarialPretrainerConfig(
	model_config=model_config,
	language_ids=ltoi,
	adv_repeat=0,
	lr=1e-4,
	beta=1e-4,
	gamma=1e-6,
	with_cuda=True,
	max_batch_size=2
)

# load datasets
# off memory streams dataset from file (fast initialization, benefits from multiprocessing)
# reads data sequentially from file only, shuffle does nothing
train_ar_raw = LanguageDataset('ar', "./example_data/train.ar.txt",
		tokenizer, seq_len, on_memory=False)
train_bg_raw = LanguageDataset('bg', "./example_data/train.bg.txt",
		tokenizer, seq_len, on_memory=False)
train_de_raw = LanguageDataset('de', "./example_data/train.de.txt",
		tokenizer, seq_len, on_memory=False)
train_en_raw = LanguageDataset('en', "./example_data/train.en.txt",
		tokenizer, seq_len, on_memory=False)

test_ar_raw = LanguageDataset('ar', "./example_data/test.ar.txt",
		tokenizer, seq_len, on_memory=False)
test_bg_raw = LanguageDataset('bg', "./example_data/test.bg.txt",
		tokenizer, seq_len, on_memory=False)
test_de_raw = LanguageDataset('de', "./example_data/test.de.txt",
		tokenizer, seq_len, on_memory=False)
test_en_raw = LanguageDataset('en', "./example_data/test.en.txt",
		tokenizer, seq_len, on_memory=False)

adversary_raw = DiscriminatorDataset("./example_data/ar-bg-de-en.txt",
		tokenizer, ltoi, seq_len, on_memory=False)

train_ar_data = DataLoader(train_ar_raw, batch_size=32, num_workers=4)
train_bg_data = DataLoader(train_bg_raw, batch_size=32, num_workers=4)
train_de_data = DataLoader(train_de_raw, batch_size=32, num_workers=4)
train_en_data = DataLoader(train_en_raw, batch_size=32, num_workers=4)

test_ar_data = DataLoader(test_ar_raw, batch_size=8)
test_bg_data = DataLoader(test_bg_raw, batch_size=8)
test_de_data = DataLoader(test_de_raw, batch_size=8)
test_en_data = DataLoader(test_en_raw, batch_size=8)

adversary_data = DataLoader(adversary_raw, batch_size=64, num_workers=16)

train_data = {
	'ar': train_ar_data,
	'bg': train_bg_data,
	'de': train_de_data,
	'en': train_en_data,
	'adversary': adversary_data
}

test_data = {
	'ar': test_ar_data,
	'bg': test_bg_data,
	'de': test_de_data,
	'en': test_en_data,
}

print({key: len(value) for key, value in train_data.items()})
print({key: len(value) for key, value in test_data.items()})

# initialize model and trainer
try:
	# try restoring from checkpoint
	test_trainer, best_epoch = AdversarialPretrainer.load_checkpoint(
			os.path.join(save_path, 'best.model.state'), MultilingualBert, train_data, test_data)
	best_loss = test_trainer.test(best_epoch)
	print("current best loss: %.6f" % best_loss)
	del test_trainer
	trainer, start = AdversarialPretrainer.load_checkpoint(
			os.path.join(save_path, 'checkpoint.state'), MultilingualBert, train_data, test_data)
except FileNotFoundError:
	model = MultilingualBert(model_config)
	trainer = AdversarialPretrainer(model, trainer_config, train_data, test_data)
	start = 0
	best_epoch = 0
	best_loss = 1e9

if not os.path.isdir(save_path):
	os.mkdir(save_path)

# train model, checkpoint every 10th epoch
with open(os.path.join(save_path, "pretraining.loss.tsv"), 'w+' if start == 0 else 'a') as f:
	f.write('epoch\ttrain\ttest\n')
	for epoch in range(start, 1000):
		epoch += 1
		train_loss = trainer.train(epoch)
		test_loss = trainer.test(epoch)
		f.write("%d\t%.6f\t%.6f\n" % (epoch, train_loss, test_loss))

		trainer.save(epoch, file_path=os.path.join(save_path, "checkpoint.state"))

		if epoch % 10 == 0:
			trainer.save(epoch, os.path.join(save_path, "epoch.%d.state" % epoch))

		if test_loss < best_loss:
			best_loss = test_loss
			best_epoch = epoch
			trainer.save(epoch, os.path.join(save_path, "best.model.state"))

		print("test loss %.6f" % test_loss)

print("best loss %f at epoch %d" % (best_loss, best_epoch))

