import os, re, io
import gzip, zipfile, tarfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import pdb


class BadCorpusError(Exception):
	pass


class CorpusReader:
	"""
	Parent class for all corpus readers.
	Assumes each file in a corpus corresponds to a single document.
	Can take a directory or compressed file as input.
	"""
	def __init__(self, corpus_path):
		if os.path.isdir(corpus_path):
			self.corpus_type = 'dir'
			self.corpus = corpus_path
			self.files = [file for _, _, files in os.walk(corpus_path) for file in files]
		elif zipfile.is_zipfile(corpus_path):
			self.corpus_type = 'zip'
			self.corpus = zipfile.ZipFile(corpus_path, 'r')
			self.files = [file for file in self.corpus.namelist() if file[-1] != '/'] # not directory
		elif tarfile.is_tarfile(corpus_path):
			self.corpus_type = 'tar'
			self.corpus = tarfile.open(corpus_path, 'r:gz')
			self.files = self.corpus.getnames()
		else:
			raise BadCorpusError("corpus \"{}\" was not a directory, zip file, or tarball".format(corpus_path))

		# TODO, place temporarily extracted files in tmp folder, then delete when done

	def extract_sentences(self, filename):
		# reads file and returns a generator yielding next sentence
		raise NotImplementedError

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		return self.extract_sentences(self.files[index])

	def __iter__(self):
		for file in self.files:
			yield self.extract_sentences(file)

	def close(self):
		if self.corpus_type in {'zip', 'tar'}:
			self.corpus.close()


class EnCzWordReader(CorpusReader):
	"""
	Reader used in dataset generator programs. Reads the Czech-English Manual
	Word Alignment corpus. <https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1804>
	"""
	def __init__(self, corpus_path, language='english'):
		super().__init__(corpus_path)
		regex = re.compile(r"data/.*\.wa")
		self.files = [file for file in self.files if regex.match(file)]
		self.language = language

	def extract_sentences(self, filename):
		# print(file)
		if self.corpus_type == 'dir':
			tree = ET.parse(filename)
			root = tree.getroot()
		elif self.corpus_type == 'tar':
			file = self.corpus.extractfile(filename)
			file_string = self.fix_xml(file.read().decode('utf-8'))
			root = ET.fromstring(file_string)
		elif self.corpus_type == 'zip':
			file = self.corpus.extract(filename)
			file_string = self.fix_xml(file.read().decode('utf-8'))
			root = ET.fromstring(file_string)
		else:
			raise BadCorpusError("corpus was not a directory or tarball")

		keys = ["sentences", "alt_sentences", "text_alignment", "alt_language", "language"]
		english, czech, text_alignment = list(), list(), list()
		for s in root:
			english.append(s.find("english").text.split())
			czech.append(s.find("czech").text.split())

			alignment = s.find("sure").text
			if alignment is None:
				alignment = list()
				continue

			alignment = [pair.split('-') for pair in alignment.split()]
			if self.language == 'english':
				text_alignment.append({int(a)-1: int(b)-1 for a, b in alignment})
			else:
				text_alignment.append({int(b)-1: int(a)-1 for a, b in alignment})
		
		if self.language == 'english':
			return dict(zip(keys, [english, czech, text_alignment, 'czech', 'english']))
		else:
			return dict(zip(keys, [czech, english, text_alignment, 'english', 'czech']))

	def fix_xml(self, file_string):
		"""
		fixes bad xml endemic to corpus and potential tokenization differences
		"""
		file_string = re.sub(r"(<english.*?>)(.*)</english>\n",
				lambda m: m.group(1) + escape(m.group(2)) + "</english>\n", file_string)
		file_string = re.sub(r"(<czech.*?>)(.*)</czech>\n",
				lambda m: m.group(1) + escape(m.group(2)) + "</czech>\n", file_string)
		return file_string


class OpenSubtitlesReader(CorpusReader):
	def __init__(self, corpus_path, language):
		super().__init__(corpus_path)
		self.language = language

	def extract_sentences(self, filename):
		if self.corpus_type == 'dir':
			file = gzip.open(filename)
		elif self.corpus_type == 'tar':
			file = gzip.open(self.corpus.extractfile(filename))
		elif self.corpus_type == 'zip':
			file = gzip.open(self.corpus.extract(filename))
		else:
			raise BadCorpusError("corpus was not a directory or tarball")

		tree = ET.parse(file)
		root = tree.getroot()
		sentences = [''.join(elem.itertext()).strip() for elem in root]
		return sentences

