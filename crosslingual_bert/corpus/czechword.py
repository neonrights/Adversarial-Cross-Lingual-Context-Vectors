import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from .corpusreader import *


class EnCzWordReader(CorpusReader):
	"""
	Reader used in dataset generator programs. Reads the Czech-English Manual
	Word Alignment corpus. <https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1804>
	"""
	def __init__(self, *args, language='english', **kwargs):
		super().__init__(*args, **kwargs)
		regex = re.compile(r"data/.*\.wa")
		self.files = [file for file in self.files if regex.match(file)]
		self.language = language

	def extract_sentences(self, file):
		# print(file)
		if self.corpus_type == 'dir':
			tree = ET.parse(file)
			root = tree.getroot()
		elif self.corpus_type == 'tar':
			file = self.corpus.extractfile(file)
			file_string = EnCzWordReader.fix_xml(file.read().decode('utf-8'))
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

	@staticmethod
	def fix_xml(file_string):
		"""
		fixes bad xml endemic to corpus and potential tokenization differences
		"""
		file_string = re.sub(r"(<english.*?>)(.*)</english>\n",
				lambda m: m.group(1) + escape(m.group(2)) + "</english>\n", file_string)
		file_string = re.sub(r"(<czech.*?>)(.*)</czech>\n",
				lambda m: m.group(1) + escape(m.group(2)) + "</czech>\n", file_string)
		return file_string

