from multiprocessing import pool
from torch.utils.data import DataLoader

from crosslingual_bert import BertTokenizer, LanguageDataset, DiscriminatorDataset
from crosslingual_bert import MultilingualBert, BertConfig
from crosslingual_bert import AdversarialPretrainer


# initialize hyperparameters
seq_len = 180
ltoi = {'ar': 0, 'bg': 1, 'de': 2, 'en': 3}
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

if ON_MEMORY:
	# stream dataset from memory (slow initialization)
	dataset_args = [
		('ar', "./example_data/sample.ar.txt", 0),
		('bg', "./example_data/sample.bg.txt", 1),
		('de', "./example_data/sample.de.txt", 2),
		('en', "./example_data/sample.en.txt", 3),
		("./example_data/sample.ar-bg-de-en.txt", 4)
	]

	def load_data_worker(args):
		if len(args) == 3:
			return LanguageDataset(args[0], args[1], tokenizer, seq_len, position=args[2], on_memory=False)
		else:
			return DiscriminatorDataset(args[0], tokenizer, ltoi, seq_len, position=args[1], on_memory=False)

	worker_pool = pool.Pool(5)
	datasets = worker_pool.map(load_data_worker, dataset_args)
	worker_pool.close()

	# batch size chosen for even number of iterations
	for dataset in datasets:
		try:
			if dataset.language == 'ar':
				ar_data = DataLoader(dataset, batch_size=8, shuffle=True)
			elif dataset.language == 'bg':
				bg_data = DataLoader(dataset, batch_size=8, shuffle=True)
			elif dataset.language == 'de':
				de_data = DataLoader(dataset, batch_size=8, shuffle=True)
			elif dataset.language == 'en':
				en_data = DataLoader(dataset, batch_size=8, shuffle=True)
		except AttributeError:
			adversary_data = DataLoader(dataset, batch_size=64, shuffle=True)
else:
	# stream dataset from file (fast initialization, benefits from multiprocessing)
	# reads data sequentially from file only
	ar_raw = LanguageDataset('ar', "./example_data/sample.ar.txt",
			tokenizer, seq_len, on_memory=False)
	bg_raw = LanguageDataset('bg', "./example_data/sample.bg.txt",
			tokenizer, seq_len, on_memory=False)
	de_raw = LanguageDataset('de', "./example_data/sample.de.txt",
			tokenizer, seq_len, on_memory=False)
	en_raw = LanguageDataset('en', "./example_data/sample.en.txt",
			tokenizer, seq_len, on_memory=False)
	adversary_raw = DiscriminatorDataset("./example_data/sample.ar-bg-de-en.txt",
			tokenizer, ltoi, seq_len, on_memory=False)

	"""ar_data = DataLoader(ar_raw, batch_size=8, shuffle=True, num_workers=4)
	bg_data = DataLoader(bg_raw, batch_size=10, shuffle=True, num_workers=4)
	de_data = DataLoader(de_raw, batch_size=4, shuffle=True, num_workers=4)
	en_data = DataLoader(en_raw, batch_size=38, shuffle=True, num_workers=4)
	adversary_data = DataLoader(adversary_raw, batch_size=32, shuffle=True, num_workers=4)"""

	ar_data = DataLoader(ar_raw, batch_size=8, num_workers=2)
	bg_data = DataLoader(bg_raw, batch_size=8, num_workers=2)
	de_data = DataLoader(de_raw, batch_size=8, num_workers=2)
	en_data = DataLoader(en_raw, batch_size=8, num_workers=2)
	adversary_data = DataLoader(adversary_raw, batch_size=64, num_workers=16)


train_data = {
	'ar': ar_data,
	'bg': bg_data,
	'de': de_data,
	'en': en_data,
	'adversary': adversary_data
}

print({key: len(value) for key, value in train_data.items()})

# initialize model and trainer
model = MultilingualBert(ltoi, config)
trainer = AdversarialPretrainer(model,
		config, ltoi,
		train_data,
		beta=1e-2, gamma=1e-6,
		adv_repeat=5)

# train model, checkpoint every 10th epoch
best_loss = 1e9
best_epoch = 0
for epoch in range(1000):
	epoch += 1
	trainer.train(epoch)
	test_loss = trainer.test(epoch)
	if epoch % 10 == 0:
		trainer.save(epoch, directory_path="pretraining")
	if test_loss < best_loss:
		best_loss = test_loss
		best_epoch = epoch
		trainer.save(epoch, directory_path="pretraining", save_name="best.model")
	print("test loss %.6f" % test_loss)

print("best loss %f at epoch %d" % (best_loss, best_epoch))

