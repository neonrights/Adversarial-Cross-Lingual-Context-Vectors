from torch.utils.data import Dataset
import tqdm
import torch
import random
import json


class LanguageDataset(Dataset):
    """
    pytorch dataset that loads training/test dataset for a specific language
    """
    def __init__(self, corpus_path, vocab, language, seq_len,
                encoding="utf-8", corpus_lines=None, on_memory=True, prob_config=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.language = language
        
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.mask_prob = prob_config.mask_prob
        self.keep_prob = prob_config.keep_prob
        self.swap_prob = self.keep_prob + prob_config.swap_prob
        self.swap_word_language_prob = prob_config.swap_word_language_prob

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        sample = self.get_corpus_line(item)
        s1, s2, is_next = self.random_sent(sample)
        s1_random, s1_label = self.random_word(s1)
        s2_random, s2_label = self.random_word(s2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        s1 = [self.vocab.sos_index] + s1_random + [self.vocab.eos_index]
        s1 = s1_random + [self.vocab.eos_index]

        s1_label = [self.vocab.pad_index] + s1_label + [self.vocab.pad_index]
        s2_label = s2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(s1))] + [2 for _ in range(len(s2))])[:self.seq_len]
        input_ids = (s1 + s2)[:self.seq_len]
        token_labels = (s1_label + s2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(input_ids))]
        input_ids.extend(padding), token_labels.extend(padding), segment_label.extend(padding)

        output = {"input_ids": input_ids,
                  "token_labels": token_labels,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return dict((key, torch.tensor(value)) if type(value) is not str else (key, value)
        		for key, value in output.items())

    def random_word(self, sample):
    	sentence= sample['sentences']
        output_label = []

        for i, token in enumerate(sentence):
            prob = random.random()
            if prob < self.mask_prob:
                prob = random.random()
                if prob < self.keep_prob: # keep same token
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                elif prob < self.swap_prob: # swap with random (keep in language?)
                    tokens[i] = random.randrange(len(self.vocab))
                else: # swap with mask
                    tokens[i] = self.vocab.mask_index

                try:
					if random.random() < self.swap_word_language_prob:
						i = sample['text_alignment'][str(i)]
						token = sample['alt_sentences'][i]
				finally:
					output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, sample):
        split = random.randint(0, len(sample_dict['sentences'] - 1))
        s1 = dict((key, value[split]) if type(value) is list else (key, value)
        		for key, value in sample.items()) # first half of segment

        if random.random() > 0.5:
        	s2 = dict((key, value[split+1]) if type(value) is list else (key, value)
        			for key, value in sample.items()) # second half of segment
        	return s1, s2, 1
        else:
            return s1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            return json.loads(line)

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()

        sample = json.loads(line)
        split = random.randint(0, len(sample['sentences']))
        return dict((key, value[split]) if type(value) is list else (key, value)
        	for key, value in sample.items())
