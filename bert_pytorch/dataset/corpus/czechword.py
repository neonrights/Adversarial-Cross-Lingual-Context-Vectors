import re
import xml.etree.ElementTree as ET

from .corpusreader import *


def fix_unescaped_xml(file_string):
	pass # replace unescaped characters where appropriate


class EnCzWordReader(CorpusReader):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		regex = re.compile(r"data/.*\.wa")
		self.files = [file for file in self.files if regex.match(file)]

	def extract_sentences(self, file):
		print(file)
		if self.corpus_type == 'dir':
			tree = ET.parse(file)
			root = tree.getroot()
		elif self.corpus_type == 'tar':
			file = self.corpus.extractfile(file)
			file_string = file.read().decode('utf-8')
			root = ET.fromstring(file_string)
		else:
			raise BadCorpusError("corpus was not a directory, zip file, or tarball")

		columns = ["text", "alt_text", "text_alignment", "alt_language", "language"]
		sentences = list()
		for s in root:
			english = s.find("english").text.split()
			czech = s.find("czech").text.split()
			alignment = [tuple(pair.split('-')) for pair in s.find("sure").text.split()]
			data = [english, czech, alignment, "czech", "english"]

			sentences.append(dict(zip(columns, data)))

		return sentences


class CzEnWordReader(CorpusReader):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		regex = re.compile(r"data/.*\.wa")
		self.files = [file for file in self.files if regex.match(file)]

	def extract_sentences(self, file):
		if self.corpus_type == 'dir':
			tree = ET.parse(file)
			root = tree.getroot()
		elif self.corpus_type == 'tar':
			file = self.corpus.extractfile(file)
			file_string = file.read().decode('utf-8')
			root = ET.fromstring(file_string)
		else:
			raise BadCorpusError("corpus was not a directory, zip file, or tarball")

		columns = ["text", "alt_text", "text_alignment", "alt_language", "language"]
		sentences = list()
		for s in root:
			english = s.find("english").text.split()
			czech = s.find("czech").text.split()
			alignment = [tuple(pair.split('-')) for pair in s.find("sure").text.split()]
			alignment = [(b, a) for a, b in alignment]
			data = [czech, english, alignment, "english", "czech"]

			sentences.append(dict(zip(columns, data)))

		return sentences

