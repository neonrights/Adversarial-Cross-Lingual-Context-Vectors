import sys
import xml.etree.cElementTree as ET


xml_prefix = "http://www.w3.org/XML/1998/namespace"
def tmx_to_pairs(tmx_file, output_file, pair):
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

				root.clear()


if __name__ == '__main__':
	tmx_to_pairs("../data/opensubtitles/en-vi.tmx", "en-vi-opensubtitles.txt", ('en', 'vi'))
