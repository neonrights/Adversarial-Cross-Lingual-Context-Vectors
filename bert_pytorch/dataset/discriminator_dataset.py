from torch.utils.data import Dataset
import tqdm
import torch
import random
import json


class DiscriminatorDataset(Dataset):
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
        input_ids.extend(padding)

        return torch.tensor(input_ids), torch.tensor(label)

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

