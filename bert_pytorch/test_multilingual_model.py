import torch
import torch.nn as nn

from model import MultilingualModel, BERT

print("Running unit tests for MultilingualModel...")

language_ids = {'en': 0, 'cz': 1, 'fr': 2, 'ch': 3, 'jp': 4}
vocab_size = 10000
model = MultilingualModel(language_ids, BERT, vocab_size=vocab_size, hidden=128, n_layers=6, attn_heads=16, dropout=0.5)

assert len(model) == 5, "number of initialized objects is wrong"
assert model.public_model is model[0].public, "public model instances are not the same object instance"
assert model[1].public is model[2].public, "public model instances are not the same object instance"
assert model['jp'] == model[4], "got different objects for equivalent indexing"
assert model[1].private is not model[2].private, "private model instances are the same object"
print("passed all initializtion tests")

test_input = torch.randint(high=vocab_size, size=[16, 64], dtype=torch.long)
segment_label = torch.randint(high=2, size=[16, 64], dtype=torch.long)
en_output = model['en'](test_input, segment_label)
ch_output = model['ch'](test_input, segment_label)

assert en_output.size() == ch_output.size(), "got different sized output between different language models"
assert en_output.size() == (16, 64, 256), "expected output size (16, 64, 256), got {}".format(en_output.size())
assert not (en_output == ch_output).all(), "got same values for different language models"
print("passed all forward propagation tests")
