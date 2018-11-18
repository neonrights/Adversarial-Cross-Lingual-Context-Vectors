import os
import sys
import os.path as path
import argparse

from corpus import *
from dataset import *
from model import *
from trainer import *


def generate_data(args):
	parser = argparse.ArgumentParser(description="Samples sequences of sentences from a specified corpus or corpora.")
	parser.add_argument('-c', '--corpus', required=True, type=str, nargs='+', help="supported corpus or list of corpora")
	parser.add_argument('-l', '--language', type=str, default=None, help="language corpora belong to")
	parser.add_argument('-a', '--adversary', action='store_true', help="flag that corpora belong to adversary")
	parser.add_argument('--length', type=int, default=512, help="max number of tokens in a sample")
	parser.add_argument('-o', '--output', type=str, default=None, help="name of output file (default name of language or adversary)")
	parser.add_argument('-r', '--random', action='store_true', help="whether to randomly sample from each corpus")
	parser.add_argument('-s', '--samples', type=int, default=1000, help="number of random samples to draw from each corpus")
	config = parser.parse_args(args)

	corpus_names = {
		"en-cz-word-aligned": EnCzWordReader(path.join('.', sys.path[0],
			"archives/CzEnAli_1.0.tar.gz"), language='english'),
		"cz-en-word-aligned": EnCzWordReader(path.join('.', sys.path[0],
			"archives/CzEnAli_1.0.tar.gz"), language='czech')
		# add support for new corpora here
	}

	config_name = "dataset.config"
	for name in config.corpus:
		try:
			reader = corpus_names[name]		
		except KeyError:
			print("{} is an unsupported corpus, try one of:".format(config.corpus))
			for name in corpus_names:
				print("\t{}".format(name))
			exit()

		if config.adversary:
			pass			
		else:
			generator = LanguageSequenceGenerator(name, config.length)
			if config.random:
				generator.random_samples(config.samples, "tmp/temp.txt")
			else:
				generator.sequential_sample("tmp/temp.txt")



def pretrain(args):
	parser = argparse.ArgumentParser(description="Runs pretraining tasks")
	parser.add_argument('-b', "--batch", type=int, default=32, help="batch size")
	parser.add_argument('-v', "--vocab", type=str, default="vocab.pkl", help="vocab file, or output name if none exists")
	parser.add_argument("--layers", type=int, default=6, help="number of hidden layers")
	parser.add_argument("--hidden", type=int, default=384, help="dimension of hidden layer (must be even)")
	parser.add_argument("--intermediate", type=int, default=1536, help="dimension of intermediate attention layers")
	parser.add_argument("--max_seq_len", type=int, default=512, help="maximum length of sequence")
	parser.add_argument("--heads", type=int, default=12, help="number of attention heads")
	parser.add_argument("--dropout", type=int, default=0.1, help="probability of dropout")
	parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
	parser.add_argument("--loss_beta", type=float, default=1e-2, help="adversarial loss weight")
	parser.add_argument("--loss_gamma", type=float, default=1e-4, help="orthogonal distance loss weight")
	parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
	parser.add_argument("--datasets", required=True, type=str, help="file specifying language and their datasets")
	parser.add_argument("--checkpoint", type=str, default='checkpoint', help="checkpoint directory")
	parser.add_argument("--save_freq", type=int, default=10, help="frequency of save checkpoints")
	config = parser.parse_args(args)

	# load dataset arrangement

	# load or create vocabulary
	if path.isfile(config.vocab):
		vocab = JSONVocab.load_vocab(config.vocab)
	else:
		vocab = JSONVocab(datasets)
		vocab.save_vocab(config.vocab)

	# load language dataset
	language_ids = {'en': 0, 'cz': 1}

	en_dataset = LanguageDataset("test_english.txt", vocab, language='en', seq_len=128)
	cz_dataset = LanguageDataset("test_czech.txt", vocab, language='cz', seq_len=128)
	D_dataset = DiscriminatorDataset("test_discriminator.txt", vocab, language_ids, seq_len=128)

	en_dataset = DataLoader(en_dataset, batch_size=config.batch_size, shuffle=True)
	cz_dataset = DataLoader(cz_dataset, batch_size=config.batch_size, shuffle=True)
	D_dataset = DataLoader(D_dataset, batch_size=config.batch_size, shuffle=True)

	train_data = {'en': en_dataset, 'cz': cz_dataset}

	# initialize model
	config = BertConfig(vocab_size=len(vocab),
			hidden_size=config.hidden//2,
			num_hidden_layers=config.layers,
			num_attention_heads=config.head,
			intermediate_size=config.intermediate,
			max_position_embeddings=config.max_seq_len)

	model = MultilingualBERT(language_ids, BertModel, config)
	adversary = SimpleAdversary(config.hidden//2, len(language_ids))
	trainer = AdversarialPretrainer(model, adversary, len(vocab), hidden, language_ids, train_data, D_dataset, train_data, 5, beta=0.1, gamma=1e-9)

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

