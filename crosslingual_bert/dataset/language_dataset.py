import random, json, itertools
import torch, tqdm
from torch.utils.data import Dataset
import pdb

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
    def __init__(self, corpus_path, tokenizer, max_seq_len,
                encoding="utf-8", corpus_lines=None, on_memory=True, prob_config=ProbConfig()):
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
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [[self.tokenizer.tokenize_and_convert_to_ids(sentence) for sentence in line.split('\t')]
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
        s1, s2, is_next = self.random_sentences(sample)
        s1_random, s1_label = self.random_word(s1)
        s2_random, s2_label = self.random_word(s2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        s1_ids = [self.tokenizer.vocab['[CLS]']] + s1_random + [self.tokenizer.vocab['[SEP]']]
        s2_ids = s2_random + [self.tokenizer.vocab['[SEP]']]

        segment_label = ([1 for _ in range(len(s1_ids))] + [2 for _ in range(len(s2_ids))])[:self.max_seq_len]
        input_ids = (s1_ids + s2_ids)[:self.max_seq_len]
        token_labels = ([0] + s1_label + [0] + s2_label + [0])[:self.max_seq_len]

        padding = [0 for _ in range(self.max_seq_len - len(input_ids))]
        mask = [1 for _ in range(len(input_ids))] + [0 for _ in range(len(padding))]
        input_ids.extend(padding), token_labels.extend(padding), segment_label.extend(padding)

        output = {"input_ids": input_ids,
                  "token_labels": token_labels,
                  "segment_label": segment_label,
                  "is_next": is_next,
                  "mask": mask}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sample):
        ids = sample.copy()
        output_label = [0] * len(sample)

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
        max_split = 0
        token_count = 2 # extra two for sos_token and eos_token
        for sentence in sample:
            token_count += len(sentence) + 1
            if token_count >= self.max_seq_len:
                break
            max_split += 1

        if max_split == 0:
            pdb.set_trace()

        split = random.randrange(max_split)
        s1 = list(itertools.chain.from_iterable(sample[:split]))

        # shorten s1 so it is less than max seq len
        if random.random() > 0.5:
            return s1, list(itertools.chain.from_iterable(sample[split:])), 1
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

            return [self.tokenizer.tokenize_and_convert_to_ids(sentence)
                    for sentence in line.split('\t')]

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
            
            sample = [self.tokenizer.tokenize_and_convert_to_ids(sentence)
                    for sentence in line.split('\t')]

        split = random.randrange(len(sample))
        return list(itertools.chain.from_iterable(sample[:split]))


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
