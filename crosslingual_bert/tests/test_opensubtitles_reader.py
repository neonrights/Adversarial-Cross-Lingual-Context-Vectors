from corpus import OpenSubtitlesReader

print("Starting smoke test for OpenSubtitlesReader")

reader = OpenSubtitlesReader("data/opensubtitles/sample.en.tar.gz")
for file in reader.files:
	assert file.endswith('.xml.gz')

print("passed file fetching test")

for file in reader:
	assert file
	for sentence in file:
		assert type(sentence) is str
		assert sentence

print("passed sentence extraction test")
