import os
import sys
import json
import os.path as path
import argparse
from torch.utils.data import DataLoader

from corpus import *
from dataset import *
from model import *
from trainer import *


def generate_data(args):
	parser = argparse.ArgumentParser(description="Samples sequences of sentences from a specified corpus or corpora.")
	parser.add_argument('-c', '--corpus-config', required=True, type=str, help="supported corpus or list of corpora")
	parser.add_argument('--max-length', type=int, default=512, help="max number of tokens in a sample")
	parser.add_argument('-o', '--output', type=str, default='sampled_data', help="name of output directory")
	parser.add_argument('-r', '--random', action='store_true', help="whether to randomly sample from each corpus")
	parser.add_argument('-s', '--samples', type=int, default=1000, help="number of random samples to draw from each corpus")
	parser.add_argument('--adv-samples', type=int, default=1000, help="number of random samples to draw for adversary")
	config = parser.parse_args(args)

	name_to_reader = {
		"en-cz-word-aligned": EnCzWordReader(path.join('.', sys.path[0],
			"data/CzEnAli_1.0.tar.gz"), language='english'),
		"cz-en-word-aligned": EnCzWordReader(path.join('.', sys.path[0],
			"data/CzEnAli_1.0.tar.gz"), language='czech')
		# add support for new corpora here
	}

	with open(config.corpus_config, 'r') as f_in:
		corpus_config = json.load(f_in)

	# convert names to readers
	try:
		corpus_config = {key: [name_to_reader[name] for name in value] for key, value in corpus_config.items()}
	except KeyError:
		print("Unsupported corpus, try one of:")
		for key in name_to_reader:
			print("\t" + key)
		exit()

	if not path.isdir(config.output):
		os.mkdir(config.output)

	dataset_config = dict()
	for language, corpus_readers in corpus_config.items():
		dataset_config[language] = path.join(config.output, language + '.txt')
		with open(dataset_config[language], 'w+') as f_out:
			for reader in corpus_readers:
				generator = LanguageSequenceGenerator(reader, config.max_length)
				if config.random:
					# append output to file
					for sample in generator.random_samples(config.samples):
						f_out.write(sample)
				else:
					for sample in generator.sequential_samples():
						f_out.write(sample)

		print("Generated dataset for {}.".format(language))

	dataset_config['adversary'] = path.join(config.output, "adversary.txt")
	generator = DiscriminatorSequenceGenerator(corpus_config, config.max_length)
	generator.random_samples(config.samples, dataset_config['adversary'])
	print("Generated dataset for adversary.")

	with open("dataset.config", 'w+') as f_out:
		json.dump(dataset_config, f_out)


def pretrain(args):
	parser = argparse.ArgumentParser(description="Runs pretraining tasks")
	parser.add_argument('-b', "--batch", type=int, default=32, help="batch size")
	parser.add_argument('-s', "--shuffle", action="store_true", help="shuffle batches")
	parser.add_argument('-v', "--vocab", type=str, default="vocab.pkl", help="vocab file, or output name if none exists")
	parser.add_argument("--train-data", required=True, type=str, help="file specifying training data for each language")
	parser.add_argument("--test-data", required=True, type=str, help="file specifying test data for each language")
	parser.add_argument("--layers", type=int, default=6, help="number of hidden layers")
	parser.add_argument("--hidden", type=int, default=384, help="dimension of hidden layer (must be even)")
	parser.add_argument("--intermediate", type=int, default=1536, help="dimension of intermediate attention layers")
	parser.add_argument("--max-seq-len", type=int, default=512, help="maximum length of sequence")
	parser.add_argument("--heads", type=int, default=12, help="number of attention heads")
	parser.add_argument("--dropout", type=int, default=0.1, help="probability of dropout")
	parser.add_argument("--learning-rate", type=float, default=1e-4, help="learning rate")
	parser.add_argument("--repeat-adv", type=int, default=5, help="number of times adversary is trained every epoch")
	parser.add_argument("--loss-beta", type=float, default=1e-2, help="adversarial loss weight")
	parser.add_argument("--loss-gamma", type=float, default=1e-4, help="orthogonal distance loss weight")
	parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
	parser.add_argument("--checkpoint", type=str, default='checkpoint', help="checkpoint directory")
	parser.add_argument("--save-freq", type=int, default=10, help="frequency of save checkpoints")
	parser.add_argument("--log-freq", type=int, default=1000, help="frequency of log of metrics")
	parser.add_argument("--gpu", action="store_true", help="use gpu to train")
	config = parser.parse_args(args)

	# load dataset arrangement
	with open(config.train_data, 'r') as f_in:
		train_files = json.load(f_in)

	with open(config.test_data, 'r') as f_in:
		test_files = json.load(f_in)

	# load or create vocabulary
	if path.isfile(config.vocab):
		print("Loading vocabulary...")
		vocab = JSONVocab.load_vocab(config.vocab)
	else:
		print("Building vocabulary...")
		vocab = JSONVocab(train_files.values())
		vocab.save_vocab(config.vocab)

	languages = [language for language in train_files if language != "adversary"]
	language_ids = {language: i for i, language in enumerate(languages)}

	# load language dataset
	train_datasets = dict((key, LanguageDataset(value, vocab, language=key, seq_len=config.max_seq_len))
			if key != "adversary" else (key, DiscriminatorDataset(value, vocab, language_ids, seq_len=config.max_seq_len))
			for key, value in train_files.items())

	test_datasets = dict((key, LanguageDataset(value, vocab, language=key, seq_len=config.max_seq_len))
			if key != "adversary" else (key, DiscriminatorDataset(value, vocab, language_ids, seq_len=config.max_seq_len))
			for key, value in train_files.items())

	# wrap each dataset in a dataloader
	train_datasets = {key: DataLoader(value, batch_size=config.batch, shuffle=config.shuffle) for key, value in train_datasets.items()}
	test_datasets = {key: DataLoader(value, batch_size=config.batch, shuffle=config.shuffle) for key, value in test_datasets.items()}

	# initialize model
	bert_config = BertConfig(
			vocab_size=len(vocab),
			hidden_size=config.hidden//2,
			num_hidden_layers=config.layers,
			num_attention_heads=config.heads,
			intermediate_size=config.intermediate,
			max_position_embeddings=config.max_seq_len)

	model = MultilingualBERT(language_ids, BertModel, bert_config)
	adversary = SimpleAdversary(config.hidden//2, len(language_ids))
	trainer = AdversarialPretrainer(
			multilingual_model=model,
			adversary_model=adversary,
			vocab_size=len(vocab),
			hidden_size=config.hidden,
			languages=language_ids,
			train_data=train_datasets,
			test_data=test_datasets,
			adv_repeat=config.repeat_adv,
			beta=config.loss_beta,
			gamma=config.loss_gamma,
			with_cuda=config.gpu,
			log_freq=config.log_freq)

	best_loss = 1e9
	for epoch in range(1000):
		trainer.train(epoch)
		test_loss = trainer.test(epoch)

		# perform checkpoints
		if test_loss < best_loss:
			trainer.save(epoch, directory="best")

		if (epoch+1) % config.save_freq == 0:
			trainer.save(epoch)


def evaluate(args):
	parser = argparse.ArgumentParser(description="Runs evaluation script for a fully trained model.")
	parser.add_argument('-m', "--model", required=True, type=str, help="directory of saved model")
	parser.add_argument('-t', "--test", type=str, default=None, help="test to run, fun all tests by default")
	parser.add_argument('-o', "--output", type=str, default='tests', help="output directory for test results")
	config = parser.parse_args(args)

	raise NotImplementedError


if __name__ == '__main__':
	try:
		if sys.argv[1] == "corpus":
			generate_data(sys.argv[2:])
		elif sys.argv[1] == "pretrain":
			pretrain(sys.argv[2:])
		elif sys.argv[1] == "evaluate":
			evaluate(sys.argv[2:])
		else:
			print("{} is an invalid module. Try corpus, pretrain, or evaluate.\
					For help with a module, invoke it.".format(sys.argv[1]))
	except IndexError:
		print("No module specified. Try corpus, pretrain, or evaluate.\
					For help with a module, invoke it.")

