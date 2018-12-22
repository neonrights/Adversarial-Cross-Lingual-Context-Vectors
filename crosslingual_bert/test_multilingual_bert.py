import torch
import torch.nn as nn

from model import MultilingualBert, BertModel, BertConfig
import pdb

print("Running unit tests for MultilingualModel...")

language_ids = {'en': 0, 'cz': 1, 'fr': 2, 'zh': 3, 'jp': 4}
vocab_size = 1000
batch_size = 16
seq_len = 64

config = BertConfig(vocab_size,
		hidden_size=100,
		num_hidden_layers=3,
		num_attention_heads=4,
		intermediate_size=200,
		max_position_embeddings=128)
model = MultilingualBert(language_ids, config)

assert len(model.private) == 5, "number of initialized objects is wrong"
assert model.private.keys() == language_ids.keys()
print("passed all initialization tests")

test_input = torch.randint(high=vocab_size, size=[batch_size, seq_len], dtype=torch.long)
segment_label = torch.randint(high=2, size=[batch_size, seq_len], dtype=torch.long)
en_hidden, en_pooled = model('en', test_input, segment_label)
zh_hidden, zh_pooled = model('zh', test_input, segment_label)

assert len(en_hidden) == len(zh_hidden)
assert en_hidden[-1].size() == zh_hidden[-1].size(), "got different sized output between different language models"
assert en_hidden[-1].size() == (batch_size, seq_len, 2*config.hidden_size), \
		"expected output size {}, got {}".format((batch_size, seq_len, 2*config.hidden_size), en_hidden[-1].size())
assert not (en_pooled == zh_pooled).all(), "got same values for different language models"
print("passed all forward propagation tests")

for param in model.parameters():
	assert isinstance(param, (torch.Tensor, nn.parameter.Parameter))

print("passed parameter fetching method test")

# TODO test freeze methods
