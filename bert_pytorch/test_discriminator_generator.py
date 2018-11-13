from corpus import EnCzWordReader, DiscriminatorSequenceGenerator


enwr = EnCzWordReader("./archives/CzEnAli_1.0.tar.gz", language='english')
czwr = EnCzWordReader("./archives/CzEnAli_1.0.tar.gz", language='czech')
msg = DiscriminatorSequenceGenerator({'en': enwr, 'cz': czwr}, 512)
msg.random_samples(1000, "discriminator.txt")
