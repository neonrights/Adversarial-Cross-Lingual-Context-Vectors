import torch
import torch.nn as nn

from model import MultilingualBert, MultilingualTranslator, BertConfig
import pdb

print("Running unit tests for MultilingualModel...")

language_ids = {'en': 0, 'cz': 1, 'fr': 2, 'ch': 3, 'jp': 4}
vocab_size = 100000
batch_size = 16
seq_len = 64

config = BertConfig(vocab_size,
		hidden_size=100,
		num_hidden_layers=3,
		num_attention_heads=4,
		intermediate_size=200,
		max_position_embeddings=128)

embedder = MultilingualBert(language_ids, config)
model = MultilingualTranslator(embedder, 'en', config)
assert config.hidden_size == 100
assert config.intermediate_size == 200
print("passed all initialization tests")

encoder_input = torch.randint(high=vocab_size, size=[batch_size, seq_len], dtype=torch.long)
decoder_input = torch.randint(high=vocab_size, size=[batch_size, seq_len], dtype=torch.long)
en_output = model('en', encoder_input, decoder_input)
ch_output = model('ch', encoder_input, decoder_input)

assert en_output.size() == ch_output.size(), "got different sized output between different language models"
assert en_output.size() == (batch_size, vocab_size), "expected output size (16, 64, 256), got {}".format(en_output.size())
assert not (en_output == ch_output).all(), "got same values for different language models"
print("passed all forward propagation tests")

encoder_mask = torch.randint(high=2, size=[batch_size, seq_len], dtype=torch.uint8)
decoder_mask = torch.randint(high=2, size=[batch_size, seq_len], dtype=torch.uint8)
en_output = model('en', encoder_input, decoder_input, encoder_mask, decoder_mask)
ch_output = model('ch', encoder_input, decoder_input, encoder_mask, decoder_mask)

assert en_output.size() == ch_output.size(), "got different sized output between different language models"
assert en_output.size() == (batch_size, vocab_size), "expected output size (16, 64, 256), got {}".format(en_output.size())
assert not (en_output == ch_output).all(), "got same values for different language models"
print("passed all forward propagation with masking tests")

# test parameters
for param in model.parameters():
	assert isinstance(param, (torch.Tensor, nn.parameter.Parameter))

print("passed parameter fetching tests")

