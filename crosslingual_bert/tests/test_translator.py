import torch

from model import TranslatorModel, BertConfig
import pdb

print("Starting unittests for translator model...")

vocab_size = 10000
seq_len = 4
batch_size = 5

config = BertConfig(vocab_size=vocab_size)
model = TranslatorModel(config)
print("passed initialization test")

test_encoder = torch.rand(batch_size, seq_len, config.hidden_size)
test_decoder = torch.rand(batch_size, seq_len, config.hidden_size)
logits = model(test_encoder, test_decoder)

assert logits.shape == (batch_size, vocab_size)

print("passed forward propagation tests")

encoder_mask = torch.randint(high=2, size=(batch_size, seq_len), dtype=torch.long)
decoder_mask = torch.randint(high=2, size=(batch_size, seq_len), dtype=torch.long)

logits = model(test_encoder, test_decoder, encoder_mask, decoder_mask)

assert logits.shape == (batch_size, vocab_size)

print("passed masking tests")
