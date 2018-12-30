import random
import json
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm


class DiscriminatorDataset(Dataset):
    """
    pytorch dataset that loads training data for model discriminator
    """
    def __init__(self, corpus_path, tokenizer, language_ids, max_seq_len,
                encoding="utf-8", corpus_lines=None, on_memory=True, position=0):
        self.tokenizer = tokenizer
        self.language_ids = language_ids
        self.max_seq_len = max_seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                self.corpus_lines = 0
                for _ in tqdm.tqdm(f, desc="Loading Adversary Dataset", total=corpus_lines, position=position):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = []
                for line in tqdm.tqdm(f, desc="Loading Adversary Dataset", total=corpus_lines, position=position):
                    sentences = line.split('\t')
                    if sentences and  len(sentences) > 2:
                        language = sentences[0]
                        sentences = [np.array(self.tokenizer.tokenize_and_convert_to_ids(sentence), dtype=np.long)
                                for sentence in sentences[1:]]
                        self.lines.append((self.language_ids[language], sentences))

                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = self._get_file_generator()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        label, input_ids = self.get_corpus_line(item)
        assert len(input_ids) > 0

        input_ids = np.hstack((self.tokenizer.vocab['[CLS]'], input_ids, self.tokenizer.vocab['[SEP]']))[:self.max_seq_len]
        if self.max_seq_len - len(input_ids) > 0:
            padding = np.zeros(self.max_seq_len - len(input_ids), dtype=np.long)
            mask = np.append(np.ones_like(input_ids), np.zeros_like(padding))
            input_ids = np.append(input_ids, padding)
        else:
            mask = np.ones(self.max_seq_len, dtype=np.long)

        return {"input_ids": torch.tensor(input_ids),
                "language_label": torch.tensor(label),
                "mask": torch.tensor(mask)}

    def get_corpus_line(self, item):
        if self.on_memory:
            label, sentences = self.lines[item]
            split = random.randrange(1, len(sentences))
            return label, np.hstack(sentences[split:])
        else:
            sentences = None # keep fetching lines until suitable one is found
            while sentences is None or len(sentences) < 2:
                line = next(self.file)
                if line is None:
                    self.file = self._get_file_generator()
                    line = next(self.file)

                sentences = line.split('\t')

            language = sentences[0]
            sentences = [np.array(self.tokenizer.tokenize_and_convert_to_ids(sentence), dtype=np.long)
                    for sentence in sentences[1:]]
            split = 0 if len(sentences) == 1 else random.randrange(len(sentences)-1)
            return self.language_ids[language], np.hstack(sentences[split:])

    def _get_file_generator(self):
        """safely read file"""
        with open(self.corpus_path, 'r') as f:
            for line in f:
                yield line


class DiscriminatorJSONDataset(Dataset):
    """
    pytorch dataset that loads training data for model discriminator
    """
    def __init__(self, corpus_path, vocab, language_ids, seq_len,
                encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.language_ids = language_ids
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

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
        line = self.get_corpus_line(item)
        label = self.language_ids[line['language']]
        
        input_ids = [self.vocab.stoi.get(token, self.vocab.unk_index)
            for sentence in line['sentences'] for token in sentence]

        input_ids = [self.vocab.sos_index] + input_ids + [self.vocab.eos_index]
        input_ids = input_ids[:self.seq_len]
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(input_ids))]
        mask = [1 for _ in range(len(input_ids))] + [0 for _ in range(len(padding))]
        input_ids.extend(padding)

        return {"input_ids": torch.tensor(input_ids),
                "language_label": torch.tensor(label),
                "mask": torch.tensor(mask)}

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

