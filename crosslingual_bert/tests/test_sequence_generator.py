from corpus import SequenceGenerator, OpenSubtitlesReader

reader = OpenSubtitlesReader("data/opensubtitles/sample.en.tar.gz")
sampler = SequenceGenerator(reader, max_seq_len=512)

sampler.sample_from_all('all.sample.en.txt')
sampler.random_samples(1000, 'random.sample.en.txt')
