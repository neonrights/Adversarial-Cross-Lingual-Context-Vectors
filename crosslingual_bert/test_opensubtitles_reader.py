import unittest
import unittest.mock as mock

from corpus import OpenSubtitlesReader

print("Starting smoke test for OpenSubtitlesReader")

reader = OpenSubtitlesReader("data/opensubtitles/sample.en.tar.gz", 'en')
for file in reader.files:
	assert file.endswith('.xml.gz')

print("passed file fetching test")

for file in reader:
	for sentence in file:
		assert type(sentence) is str
		assert sentence

print("passed sentence extraction test")
