from dataset import JSONVocab
import pdb

print("Running smoke tests for JSONVocab...")

vocab = JSONVocab(["test_discriminator.txt", "test_language.txt"])
assert len(vocab.itos) > 5
print("passed word frequency test")

# assert class variables and functions
vocab.save_vocab("test_vocab.pkl")
loaded_vocab = JSONVocab.load_vocab("test_vocab.pkl")
# rerun previous assertions
assert len(loaded_vocab.itos) == len(vocab.itos)
print("passed save and load test")
