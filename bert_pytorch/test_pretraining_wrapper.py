import torch
import torch.nn as nn
import pdb

from model import MultilingualModel, PretrainingWrapper, BERT


print("Running unit tests for PretrainingWrapper...")

language_ids = {'en': 0, 'cz': 1, 'fr': 2, 'ch': 3, 'jp': 4}
vocab_size = 10000
model = MultilingualModel(language_ids, BERT, vocab_size=vocab_size, hidden=128, n_layers=6, attn_heads=16, dropout=0.5)
pretrain_model = PretrainingWrapper(model, nn.Linear(256, 5), 256, vocab_size)

assert len(model) == 5, "number of initialized objects is wrong"
assert pretrain_model[0] is pretrain_model['en']
assert pretrain_model[0] is not pretrain_model[1]
print("passed all initialization tests")

test_input = torch.randint(high=vocab_size, size=[16, 64], dtype=torch.long)
segment_label = torch.randint(high=2, size=[16, 64], dtype=torch.long)
en_output = pretrain_model['en'](test_input, segment_label)
ch_output = pretrain_model['ch'](test_input, segment_label)

assert en_output is not ch_output
for en_item, ch_item in zip(en_output, ch_output):
	assert en_item.size() == ch_item.size(), "got different sized output between di{fferent language models"

assert en_output[0].size() == (16, 64, vocab_size), "expected output size (16, 64, {}), got {}".format(vocab_size, en_output[0].size())
assert en_output[1].size() == (16, 2), "expected output size (16, 2), got {}".format(en_output[1].size())
assert en_output[2].size() == (16, 5), "expected output size (16, 5), got {}".format(en_output[2].size())
assert en_output[3].size() == (), "expected scalar value, got tensor of size {}".format(en_output[3].size())
print("passed all forward propagation tests")
