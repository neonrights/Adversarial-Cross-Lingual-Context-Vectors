import os
import torch
import torch,multiprocessing as mp

from torch.utils.data import DataLoader

from crosslingual_bert.dataset import BertTokenizer, LanguageDataset, DiscriminatorDataset
from crosslingual_bert.model import MultilingualBert, MultilingualConfig
from crosslingual_bert.trainer import DistributedAdversarialPretrainer, AdversarialPretrainerConfig


# initialize hyperparameters
save_path = "large-1"
seq_len = 192 # XNLI max sequence length with wordpiece tokenization is 167
#ltoi = {'ar': 0, 'bg': 1, 'de': 2, 'en': 3}
ltoi = {'ar': 0, 'bg': 1}
tokenizer = BertTokenizer("./example_data/bert-base-multilingual-cased-vocab.txt")
gpu_count = torch.cuda.device_count()

model_config = MultilingualConfig(
	languages=ltoi,
	vocab_size=len(tokenizer.vocab),
    #hidden_size,
    #intermediate_size=768
)

trainer_config = AdversarialPretrainerConfig(
	model_config=model_config,
	language_ids=ltoi,
	adv_repeat=0,
	lr=1e-4,
	beta=1e-4,
	gamma=1e-6,
    train_freq=4,
    share_file="./share_file"
)

# load datasets
# off memory streams dataset from file (fast initialization, benefits from multiprocessing)
# reads data sequentially from file only, shuffle does nothing
train_ar_raw = LanguageDataset('ar', "./data/train.ar.txt",
		tokenizer, seq_len, on_memory=False)
train_bg_raw = LanguageDataset('bg', "./data/train.bg.txt",
		tokenizer, seq_len, on_memory=False)
train_de_raw = LanguageDataset('de', "./data/train.de.txt",
		tokenizer, seq_len, on_memory=False)
train_en_raw = LanguageDataset('en', "./data/train.en.txt",
		tokenizer, seq_len, on_memory=False)

test_ar_raw = LanguageDataset('ar', "./data/test.ar.txt",
		tokenizer, seq_len, on_memory=False)
test_bg_raw = LanguageDataset('bg', "./data/test.bg.txt",
		tokenizer, seq_len, on_memory=False)
test_de_raw = LanguageDataset('de', "./data/test.de.txt",
		tokenizer, seq_len, on_memory=False)
test_en_raw = LanguageDataset('en', "./data/test.en.txt",
		tokenizer, seq_len, on_memory=False)

adversary_raw = DiscriminatorDataset("./data/shuf.ar-bg-de-en.txt",
		tokenizer, ltoi, seq_len, on_memory=False)

train_ar_data = DataLoader(train_ar_raw, batch_size=4,
        num_workers=2, drop_last=True, pin_memory=True)
train_bg_data = DataLoader(train_bg_raw, batch_size=4,
        num_workers=2, drop_last=True, pin_memory=True)
train_de_data = DataLoader(train_de_raw, batch_size=4,
        num_workers=2, drop_last=True, pin_memory=True)
train_en_data = DataLoader(train_en_raw, batch_size=4,
        num_workers=2, drop_last=True, pin_memory=True)

test_ar_data = DataLoader(test_ar_raw, batch_size=4, pin_memory=True)
test_bg_data = DataLoader(test_bg_raw, batch_size=4, pin_memory=True)
test_de_data = DataLoader(test_de_raw, batch_size=4, pin_memory=True)
test_en_data = DataLoader(test_en_raw, batch_size=4, pin_memory=True)

adversary_data = DataLoader(adversary_raw, batch_size=32,
        num_workers=8, pin_memory=True)

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

mp.spawn(DistributedAdversarialPretrainer.worker, nprocs=gpu_count,
		args=(trainer_config, train_data, test_data))

