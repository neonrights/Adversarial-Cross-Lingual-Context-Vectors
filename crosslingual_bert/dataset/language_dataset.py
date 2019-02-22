import os
import os.path as path
import time
import random
import json
import itertools
from multiprocessing import Queue, Process

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset


class ProbConfig:
    """
    Data carrying class for pretraining probabilities
    """
    def __init__(self, mask_prob=0.15, keep_prob=0.1, swap_prob=0.1, swap_language_prob=0.1):
        self.mask_prob = mask_prob
        self.keep_prob = keep_prob
        self.swap_prob = swap_prob
        self.swap_language_prob = swap_language_prob


class LanguageDataset(Dataset):
    """
    pytorch dataset that loads training/test dataset for a specific language
    """
    def __init__(self, language, corpus_path, tokenizer, max_seq_len,
                encoding="utf-8", prob_config=ProbConfig(), verbose=False):
        self.language = language
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.mask_prob = prob_config.mask_prob
        self.keep_prob = prob_config.keep_prob
        self.swap_prob = prob_config.keep_prob + prob_config.swap_prob

        # walk through directory, determine which files are adequate
        self.file_names = []
        for root, _, files in os.walk(corpus_path):
            if verbose:
                print("scanning %s" % root)
            for file in files:
                file_name = path.join(root, file)
                with open(file_name, 'r', encoding=self.encoding) as f:
                    sentences = f.readlines()

                if len(sentences) > 1 and (len(sentences[0]) + 2) < self.max_seq_len:
                    self.file_names.append(file_name)

        if verbose:
            print("processed %d samples" % len(self.file_names))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        sample = self.get_corpus_line(item)
        s1, s2, is_next = self.random_sentences(sample)
        s1_random, s1_label = self.random_word(s1)
        s2_random, s2_label = self.random_word(s2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        s1_ids = np.hstack((self.tokenizer.vocab['[CLS]'], s1_random, self.tokenizer.vocab['[SEP]']))
        s2_ids = np.hstack((s2_random, self.tokenizer.vocab['[SEP]']))

        segment_label = np.append(np.zeros_like(s1_ids), np.ones_like(s2_ids))[:self.max_seq_len]
        input_ids = np.append(s1_ids, s2_ids)[:self.max_seq_len]
        token_labels = np.hstack((0, s1_label, 0, s2_label, 0))[:self.max_seq_len]

        if self.max_seq_len - len(input_ids) > 0:
            padding = np.zeros(self.max_seq_len - len(input_ids), dtype=np.long)
            mask = np.hstack((np.ones_like(input_ids), np.zeros_like(padding)))
            input_ids = np.append(input_ids, padding)
            token_labels = np.append(token_labels, padding)
            segment_label = np.append(segment_label, padding)
        else:
            mask = np.ones(self.max_seq_len, dtype=np.long)

        output = {"input_ids": input_ids,
                  "token_labels": token_labels,
                  "segment_label": segment_label,
                  "is_next": is_next,
                  "mask": mask}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sample):
        ids = sample.copy()
        output_label = np.zeros_like(sample)

        for i in range(len(sample)):
            prob = random.random()
            if prob < self.mask_prob:
                prob = random.random()
                if prob < self.keep_prob: # keep same token
                    continue
                elif prob < self.swap_prob: # swap with random (keep in language?)
                    ids[i] = random.randrange(107, len(self.tokenizer.vocab))
                else: # swap with mask
                    ids[i] = self.tokenizer.vocab['[MASK]']

                output_label[i] = sample[i]

        return ids, output_label

    def random_sentences(self, sample):
        # ensures first section is less than max_seq_len
        assert len(sample) > 1
        max_split = 0
        token_count = 2 # extra two for sos_token and eos_token
        for sentence in sample:
            token_count += len(sentence)
            if token_count >= self.max_seq_len:
                break
            max_split += 1

        split = 1 if max_split == 1 else random.randrange(1, max_split)
        s1 = np.hstack(sample[:split])

        # shorten s1 so it is less than max seq len
        if random.random() > 0.5:
            return s1, np.hstack(sample[split:]), 1
        else:
            return s1, np.hstack(self.get_random_line()), 0

    def get_corpus_line(self, item):
        with open(self.file_names[item], 'r', encoding=self.encoding) as f:
            sentences = f.readlines()

        sentences = [np.array(self.tokenizer.tokenize_and_convert_to_ids(sentence), dtype=np.long)
                    for sentence in sentences]
        return sentences

    def get_random_line(self):
        with open(random.choice(self.file_names), 'r', encoding=self.encoding) as f:
            sentences = f.readlines()

        sample = [np.array(self.tokenizer.tokenize_and_convert_to_ids(sentence), dtype=np.long)
                for sentence in sentences]
        split = 0 if len(sample) == 1 else random.randrange(len(sample)-1)
        return sample[split:]


class LanguageStreamDataset(LanguageDataset):
    """
    pytorch dataset that loads training/test dataset for a specific language
    """
    def __init__(self, language, corpus_path, tokenizer, max_seq_len,
                encoding="utf-8", corpus_lines=None, on_memory=True, prob_config=ProbConfig(), position=0):
        self.language = language
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.mask_prob = prob_config.mask_prob
        self.keep_prob = prob_config.keep_prob
        self.swap_prob = prob_config.keep_prob + prob_config.swap_prob

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                self.corpus_lines = 0
                for _ in tqdm.tqdm(f, desc="Loading Dataset %s" % self.language, total=corpus_lines, position=position):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = []
                for line in tqdm.tqdm(f, desc="Loading Dataset %s" % self.language, total=corpus_lines, position=position):
                    sentences = line.split('\t')
                    if sentences and len(sentences) > 1:
                        sentences = [np.array(self.tokenizer.tokenize_and_convert_to_ids(sentence), dtype=np.long)
                                for sentence in sentences]
                        if (len(sentences[0]) + 2) < self.max_seq_len:
                            self.lines.append(sentences)

                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = Queue(maxsize=10000)
            self.random_file = Queue(maxsize=10000)

            # initialize slave process to safely read from file
            self.slave = Process(target=LanguageDataset.file_slave,
                    args=(self.corpus_path, self.file, self.random_file, self.corpus_lines))
            self.slave.daemon = True
            self.slave.start()

    def __len__(self):
        return self.corpus_lines

    def random_sentences(self, sample):
        # ensures first section is less than max_seq_len
        assert len(sample) > 1
        max_split = 0
        token_count = 2 # extra two for sos_token and eos_token
        for sentence in sample:
            token_count += len(sentence)
            if token_count >= self.max_seq_len:
                break
            max_split += 1

        split = 1 if max_split == 1 else random.randrange(1, max_split)
        s1 = np.hstack(sample[:split])

        # shorten s1 so it is less than max seq len
        if random.random() > 0.5:
            return s1, np.hstack(sample[split:]), 1
        else:
            return s1, np.hstack(self.get_random_line()), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item]

        line = self.file.get()
        sentences = [np.array(self.tokenizer.tokenize_and_convert_to_ids(sentence), dtype=np.long)
                    for sentence in line.split('\t')]
        if not sentences or len(sentences) < 2 or (len(sentences[0]) + 2) >= self.max_seq_len:
            return self.get_random_line(split=False)
        else:
            return sentences

    def get_random_line(self, split=True):
        if self.on_memory:
            sample = self.lines[random.randrange(len(self.lines))]
        else:
            sample = None
            while sample is None or len(sample) < 1 or (len(sample[0]) + 2) >= self.max_seq_len:
                line = self.random_file.get()
                sample = [np.array(self.tokenizer.tokenize_and_convert_to_ids(sentence), dtype=np.long)
                        for sentence in line.split('\t')]

        if split:
            split = 0 if len(sample) == 1 else random.randrange(len(sample)-1)
            return sample[split:]
        else:
            return sample


class LanguageSwapDataset(Dataset):
    """
    pytorch dataset that loads training/test dataset for a specific language
    """
    def __init__(self, corpus_path, vocab, language, seq_len,
                encoding="utf-8", corpus_lines=None, on_memory=True, prob_config=ProbConfig()):
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
        self.swap_language_prob = prob_config.swap_language_prob

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                self.corpus_lines = 0
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [json.loads(line)
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randrange(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        sample = self.get_corpus_line(item)
        s1, s2, is_next = self.random_sent(sample)
        s1_random, s1_label = self.random_word(s1)
        s2_random, s2_label = self.random_word(s2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        s1_ids = [self.vocab.sos_index] + s1_random + [self.vocab.eos_index]
        s2_ids = s2_random + [self.vocab.eos_index]

        s1_label = [self.vocab.pad_index] + s1_label + [self.vocab.pad_index]
        s2_label = s2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(s1_ids))] + [2 for _ in range(len(s2_ids))])[:self.seq_len]
        input_ids = (s1_ids + s2_ids)[:self.seq_len]
        token_labels = (s1_label + s2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(input_ids))]
        mask = [1 for _ in range(len(input_ids))] + [0 for _ in range(len(padding))]
        input_ids.extend(padding), token_labels.extend(padding), segment_label.extend(padding)
        
        output = {"input_ids": input_ids,
                  "token_labels": token_labels,
                  "segment_label": segment_label,
                  "is_next": is_next,
                  "mask": mask}

        return dict((key, torch.tensor(value)) if type(value) is not str else (key, value)
                for key, value in output.items())

    def random_word(self, sample):
        tokens = sample['sentences'].copy()
        output_label = list()

        for i, token in enumerate(tokens):
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
                    if random.random() < self.swap_language_prob:
                        i = sample['text_alignment'][str(i)]
                        token = sample['alt_sentences'][i]
                except KeyError:
                    pass # no aligned token
                finally:
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, sample):
        split = random.randrange(len(sample['sentences']) - 1)
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
            sample = self.lines[random.randrange(len(self.lines))]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                    self.random_file.__next__()
                line = self.random_file.__next__()
            
            sample = json.loads(line)
        
        split = random.randrange(len(sample['sentences']))
        return dict((key, value[split]) if type(value) is list else (key, value)
            for key, value in sample.items())
