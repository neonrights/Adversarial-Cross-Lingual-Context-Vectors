from corpus import EnCzWordReader, LanguageSequenceGenerator


cwr = EnCzWordReader("./archives/CzEnAli_1.0.tar.gz")
msg = LanguageSequenceGenerator(cwr, 512)
msg.random_samples(1000, "test_language.txt")