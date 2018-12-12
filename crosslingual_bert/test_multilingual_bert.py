import torch
import torch.nn as nn

from model import MultilingualBert, BertModel, BertConfig
import pdb

print("Running unit tests for MultilingualModel...")

language_ids = {'en': 0, 'cz': 1, 'fr': 2, 'ch': 3, 'jp': 4}
vocab_size = 1000
batch_size = 16
seq_len = 64

config = BertConfig(vocab_size)
model = MultilingualBert(language_ids, BertModel, config)
components = model.get_components()

assert len(model) == 5, "number of initialized objects is wrong"
assert components["public"] is model[0].public, "public model instances are not the same object instance"
assert model[1].public is model[2].public, "public model instances are not the same object instance"
assert model['jp'] is model[4], "got different objects for equivalent indexing"
assert model[1].private is not model[2].private, "private model instances are the same object"
print("passed all initialization tests")

test_input = torch.randint(high=vocab_size, size=[batch_size, seq_len], dtype=torch.long)
segment_label = torch.randint(high=2, size=[batch_size, seq_len], dtype=torch.long)
en_hidden, en_pooled = model['en'](test_input, segment_label)
ch_hidden, ch_pooled = model['ch'](test_input, segment_label)

assert len(en_hidden) == len(ch_hidden)
assert en_hidden[-1].size() == ch_hidden[-1].size(), "got different sized output between different language models"
assert en_hidden[-1].size() == (batch_size, seq_len, 2*config.hidden_size), \
		"expected output size {}, got {}".format((batch_size, seq_len, 2*config.hidden_size), en_hidden[-1].size())
assert not (en_pooled == ch_pooled).all(), "got same values for different language models"
print("passed all forward propagation tests")

for param in model.parameters():
	assert isinstance(param, (torch.Tensor, nn.parameter.Parameter))

print("passed parameter fetching method test")
