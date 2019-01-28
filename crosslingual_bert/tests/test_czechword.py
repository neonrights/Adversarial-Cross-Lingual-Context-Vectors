from corpus import EnCzWordReader

print("Running smoke tests")
cwr = EnCzWordReader("./data/CzEnAli_1.0.tar.gz")
foo = cwr[0]
assert type(foo) is dict
assert type(foo["sentences"]) is list
assert set(foo.keys()) == {"sentences", "alt_sentences", "text_alignment", "alt_language", "language"}
print("passed smoke tests")