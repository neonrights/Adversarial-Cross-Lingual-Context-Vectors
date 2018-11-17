from corpus import EnCzWordReader, LanguageSequenceGenerator


print("Running smoke tests for language generator")

enwr = EnCzWordReader("./archives/CzEnAli_1.0.tar.gz")
msg = LanguageSequenceGenerator(enwr, 512)
msg.random_samples(1000, "test_english.txt")

czwr = EnCzWordReader("./archives/CzEnAli_1.0.tar.gz")
msg = LanguageSequenceGenerator(czwr, 512)
msg.random_samples(1000, "test_czech.txt")