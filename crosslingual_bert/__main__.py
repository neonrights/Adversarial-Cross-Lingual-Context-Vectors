import os
import sys
import os.path as path
import argparse

from corpus import *
from dataset import *
from model import *
from trainer import *


corpus_names = {
	"en-cz-word-aligned": EnCzWordReader("./archives/CzEnAli_1.0.tar.gz", language='english'),
	"cz-en-word-aligned": EnCzWordReader("./archives/CzEnAli_1.0.tar.gz", language='czech')
	# add support for new corpora here
}

def generate_data(args):
	parser = argparse.ArgumentParser(description="Samples sequences of sentences from a specified corpus or corpora.")
	parser.add_argument('-c', '--corpus', type=str, nargs='+', help="supported corpus or list of corpora")
	parser.add_argument('-l', '--language', type=str, default=None, help="language corpora belong to")
	parser.add_argument('-a', '--adversary', type=bool, action='store_true', help="flag that corpora belong to adversary")
	parser.add_argument('--length', type=int, default=512, help="max number of tokens in a sample (default 512)")
	parser.add_argument('-o', '--output', type=str, default=None, help="name of output file (default name of language or adversary)")
	parser.add_argument('-r', '--random', action='store_true', help="whether to randomly sample from each corpus")
	parser.add_argument('-s', '--samples', type=int, default=1000, help="number of random samples to draw from each corpus (default 1000)")
	config = parser.parser_args(args)

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
	parser.add_argument()

	config = parser.parser_args(args)

	vocab = JSONVocab.load_vocab("test_vocab.pkl")
	language_ids = {'en': 0, 'cz': 1}

	en_dataset = LanguageDataset("test_english.txt", vocab, language='en', seq_len=128)
	cz_dataset = LanguageDataset("test_czech.txt", vocab, language='cz', seq_len=128)
	D_dataset = DiscriminatorDataset("test_discriminator.txt", vocab, language_ids, seq_len=128)

	en_dataset = DataLoader(en_dataset, batch_size=32, shuffle=True)
	cz_dataset = DataLoader(cz_dataset, batch_size=32, shuffle=True)
	D_dataset = DataLoader(D_dataset, batch_size=32, shuffle=True)

	train_data = {'en': en_dataset, 'cz': cz_dataset}

	# initialize models
	hidden = 256
	config = BertConfig(vocab_size=len(vocab),
			hidden_size=hidden//2,
			num_hidden_layers=6,
			num_attention_heads=8,
			intermediate_size=hidden,
			max_position_embeddings=256)

	model = MultilingualBERT(language_ids, BertModel, config)
	adversary = SimpleAdversary(hidden//2, len(language_ids))
	trainer = AdversarialPretrainer(model, adversary, len(vocab), hidden, language_ids, train_data, D_dataset, train_data, 5, beta=0.1, gamma=1e-9)

	for epoch in range(1000):
		trainer.train(epoch)

		if (epoch+1) % 10 == 0:
			trainer.save(epoch)


def evaluate(args):
	raise NotImplementedError


if __name__ == '__main__':
	try:
		if sys.argv[1] == "corpus":
			generate_data(sys.argv[1:])
		elif sys.argv[1] == "pretrain":
			pretrain(sys.argv[1:])
		elif sys.argv[1] == "evaluate":
			evaluate(sys.argv[1:])
		else:
			print("{} is an invalid module. Try corpus, pretrain, or evaluate.\
					For help with a module, invoke it.".format(sys.argv[1]))
	except IndexError:
		print("No module specified. Try corpus, pretrain, or evaluate.\
					For help with a module, invoke it.")