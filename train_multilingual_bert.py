from multiprocessing import pool
from torch.utils.data import DataLoader

from crosslingual_bert import BertTokenizer, LanguageDataset, DiscriminatorDataset
from crosslingual_bert import MultilingualBert, BertConfig
from crosslingual_bert import AdversarialPretrainer
import pdb

# initialize hyperparameters
seq_len = 256
ltoi = {'ar': 0, 'bg': 1, 'de': 2, 'en': 3}
tokenizer = BertTokenizer("/home/neonrights/Documents/project/crosslingual_bert/data/bert-base-multilingual-cased-vocab.txt")
config = BertConfig(len(tokenizer.vocab),
		hidden_size=384,
		num_hidden_layers=6,
		num_attention_heads=12,
		intermediate_size=1536,
		hidden_act='gelu',
		max_position_embeddings=512)

# load datasets
"""dataset_args = [
	('ar', "/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/ar.raw.txt", 0),
	('bg', "/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/bg.raw.txt", 1),
	('de', "/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/de.raw.txt", 2),
	('en', "/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/en.raw.txt", 3),
	("/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/ar-bg-de-en.raw.txt", 4)
]

def load_data_worker(args):
	if len(args) == 3:
		return LanguageDataset(args[0], args[1], tokenizer, seq_len, on_memory=False, position=args[2])
	else:
		return DiscriminatorDataset(args[0], ltoi, tokenizer, seq_len, on_memory=False, position=args[1])

worker_pool = pool.Pool(16)
datasets = worker_pool.map(load_data_worker, dataset_args)

# batch size chosen for even number of iterations
for dataset in datasets:
	if dataset.language == 'ar':
		ar_data = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
	elif dataset.language == 'bg':
		bg_data = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
	elif dataset.language == 'de':
		de_data = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
	elif dataset.language == 'en':
		en_data = DataLoader(dataset, batch_size=38, shuffle=True, num_workers=4)
	else:
		adversary_data = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)"""

ar_raw = LanguageDataset('ar', "/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/ar.raw.txt",
		tokenizer, seq_len, on_memory=False)
bg_raw = LanguageDataset('bg', "/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/bg.raw.txt",
		tokenizer, seq_len, on_memory=False)
de_raw = LanguageDataset('de', "/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/de.raw.txt",
		tokenizer, seq_len, on_memory=False)
en_raw = LanguageDataset('en', "/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/en.raw.txt",
		tokenizer, seq_len, on_memory=False)
adversary_raw = DiscriminatorDataset("/home/neonrights/Documents/project/crosslingual_bert/data/opensubtitles/ar-bg-de-en.raw.txt",
		tokenizer, ltoi, seq_len, on_memory=False)

"""cause pytorch num_workers broken af
ar_data = DataLoader(ar_raw, batch_size=8, shuffle=True, num_workers=4)
bg_data = DataLoader(bg_raw, batch_size=10, shuffle=True, num_workers=4)
de_data = DataLoader(de_raw, batch_size=4, shuffle=True, num_workers=4)
en_data = DataLoader(en_raw, batch_size=38, shuffle=True, num_workers=4)
adversary_data = DataLoader(adversary_raw, batch_size=32, shuffle=True, num_workers=4)
"""

ar_data = DataLoader(ar_raw, batch_size=8, num_workers=4)
bg_data = DataLoader(bg_raw, batch_size=10, num_workers=4)
de_data = DataLoader(de_raw, batch_size=4, num_workers=4)
en_data = DataLoader(en_raw, batch_size=38, num_workers=8)
adversary_data = DataLoader(adversary_raw, batch_size=32, shuffle=True, num_workers=8)

train_data = {
	'ar': ar_data,
	'bg': bg_data,
	'de': de_data,
	'en': en_data,
	'adversary': adversary_data
}

# initialize model and trainer
model = MultilingualBert(ltoi, config)
trainer = AdversarialPretrainer(model, config, ltoi, train_data, adv_repeat=3)

# train model, checkpoint every 10th epoch
best_loss = 1e9
best_epoch = 0
for epoch in range(1000):
	trainer.train(epoch)
	test_loss = trainer.test(epoch)
	if (epoch+1) % 10 == 0:
		trainer.save(epoch)
	if test_loss < best_loss:
		best_loss = test_loss
		best_epoch = epoch
		trainer.save(epoch, savename="best.model")

print("best loss %f at epoch %d" % (best_loss, best_epoch))

