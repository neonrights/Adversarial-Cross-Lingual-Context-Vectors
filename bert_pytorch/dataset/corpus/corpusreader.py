import os
import zipfile
import tarfile


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

		# TODO, place temporarily extracted files in tmp folder

	def extract_sentences(self, file):
		# reads file and returns a generator yielding next sentence
		raise NotImplementedError

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		return self.extract_sentences(self.files[index])

	def __iter__(self):
		return iter(self)

	def close(self):
		if self.corpus_type in {'zip', 'tar'}:
			self.corpus.close()

