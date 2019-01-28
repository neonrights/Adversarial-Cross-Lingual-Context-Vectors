import random

samples = 1e9
languages = ['ar', 'bg', 'de', 'en']
for language in languages:
	with open('data/opensubtitles/%s.raw.txt' % language, 'r') as f_in:
		count = 0
		for line in f_in:
			count += 1
	samples = min(count, samples)

for language in languages:
	with open('data/opensubtitles/%s.raw.txt' % language, 'r') as f_in, open('%s.txt' % language, 'w+') as f_out:
		lines = iter(f_in)
		for _ in range(samples):
			try:
				line = next(lines).strip()
				while not line:
					line = next(lines).strip()
				f_out.write(line + '\n')
			except StopIteration:
				break