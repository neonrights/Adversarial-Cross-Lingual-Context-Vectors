from dataset import JSONVocab
import pdb

vocab = JSONVocab(["test_discriminator.txt", "test_language.txt"])
# assert class variables and functions
vocab.save_vocab("test_vocab.pkl")
vocab = JSONVocab.load_vocab("test_vocab.pkl")
# rerun previous assertions
