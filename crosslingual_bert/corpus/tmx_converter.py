import sys
import argparse
import xml.etree.cElementTree as ET

"http://www.w3.org/XML/1998/namespace"
def tmx_to_pairs(tmx_file, output_file, pair):
	"""Finds bilingual pairs of sentences in a Translation Memory eXchange (.tmx) file.
	Writes language pairs found as a tsv file.

	tmx_file -- (str|) name of .tmx file
	output_file -- (str) name of desired output file
	pair -- languages to write to file
	"""
	with open(output_file, 'w+') as f_out:
		tmx_tree = iter(ET.iterparse(tmx_file, events=("start", "end")))
		_, root = next(tmx_tree)
		f_out.write("{}\t{}\n".format(*pair))

		trans = {}
		text = None
		processed = 0
		for event, elem in tmx_tree:
			if event == 'end':
				if elem.tag == 'tu':
					assert trans.keys() == set(pair)
					# write tokenized pair, reset pair recorded
					f_out.write("{}\t{}\n".format(*(trans[lang] for lang in pair)))
					processed += 1
					trans = {}

				elif elem.tag == 'tuv':
					lang = elem.get('{http://www.w3.org/XML/1998/namespace}lang')
					assert lang is not None
					trans[lang] = text
				elif elem.tag =='seg':
					text = elem.text

				elem.clear()

			root.clear()

		del tmx_tree


if __name__ == '__main__':
	"""parser = argparse.ArgumentParser(description="Find bilingual pairs in a .tmx file")
	parser.add_argument('--tmx', '-t', type=str, required=True, help="tmx file")
	parser.add_argument('--output', '-o', type=str, required=True, help="output file")
	parser.add_argument('--languages' '-l', type=str, nargs='+', required=True, help="expected language pairs")
	"""
	print('el')
	tmx_to_pairs("../data/opensubtitles/el-en.tmx", "../data/opensubtitles/en-el.tsv", ('en', 'el'))
	print('es')
	tmx_to_pairs("../data/opensubtitles/en-es.tmx", "../data/opensubtitles/en-es.tsv", ('en', 'es'))
	print('fr')
	tmx_to_pairs("../data/opensubtitles/en-fr.tmx", "../data/opensubtitles/en-fr.tsv", ('en', 'fr'))
	print('hi')
	tmx_to_pairs("../data/opensubtitles/en-hi.tmx", "../data/opensubtitles/en-hi.tsv", ('en', 'hi'))
	print('ru')
	tmx_to_pairs("../data/opensubtitles/en-ru.tmx", "../data/opensubtitles/en-ru.tsv", ('en', 'ru'))
	print('sw')
	tmx_to_pairs("../data/opensubtitles/en-sw.tmx", "../data/opensubtitles/en-sw.tsv", ('en', 'sw'))
