import torch
import torch.nn as nn

from model import MultilingualBert, MultilingualTranslator,\
		BertConfig, BertModel, TranslatorModel

print("Running unit tests for MultilingualModel...")

language_ids = {'en': 0, 'cz': 1, 'fr': 2, 'ch': 3, 'jp': 4}
vocab_size = 1000
batch_size = 16
seq_len = 64

config = BertConfig(vocab_size,
		hidden_size=300,
		num_hidden_layers=3,
		intermediate_size=450)
embedder = MultilingualBert(language_ids, BertModel, config)

config.hidden_size *= 2
config.intermediate_size *= 2
translator = TranslatorModel(config)
model = MultilingualTranslator(embedder, translator, 'en')

assert len(model) == 5, "number of initialized objects is wrong"
assert model.language_translators[0] is model['en']
assert model['ch'] is model[3]
assert model['en'].translator_model is model.models['translator']
print("passed all initialization tests")


encoder_input = torch.randint(high=vocab_size, size=[batch_size, seq_len], dtype=torch.long)
decoder_input = torch.randint(high=vocab_size, size=[batch_size, seq_len], dtype=torch.long)
en_output = model['en'](encoder_input, decoder_input)
ch_output = model['ch'](encoder_input, decoder_input)

assert en_output.size() == ch_output.size(), "got different sized output between different language models"
assert en_output.size() == (batch_size, vocab_size), "expected output size (16, 64, 256), got {}".format(en_output.size())
assert not (en_output == ch_output).all(), "got same values for different language models"
print("passed all forward propagation tests")
