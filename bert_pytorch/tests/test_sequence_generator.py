from sequence_generator import *
from corpus import *


cwr = EnCzWordReader("/home/neonrights/Documents/Data Archives/CzEnAli_1.0.tar.gz")
msg = MonolingualSequenceGenerator(cwr, 512)
msg.random_samples(1000, "samples.txt")
