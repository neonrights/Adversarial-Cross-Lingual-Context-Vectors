from czechword import EnCzWordReader


if __name__ == '__main__':
	print("Running smoke tests")
	cwr = EnCzWordReader("/home/neonrights/Documents/Data Archives/CzEnAli_1.0.tar.gz")
	foo = cwr[0]
	assert type(foo) is list
	assert type(foo[0]) is dict
	assert set(foo[0].keys()) == {"text", "alt_text", "text_alignment", "alt_language", "language"}
	print("passed smoke tests")