import random


class SmoothedRandomSampler:
    """Randomly sample from datasets in proportion to their size downsampled
    by a factor.
    """
    def __init__(self, datasets, factor=0.7, seed=None):
        self.datasets = datasets
        self.factor = factor

        probs = {key: len(value)**factor for key, value in datasets.items()}
        total = sum(probs.values())
        self.probs = {name: prob / total for name, prob in probs.items()}
        min_name = min(datasets, key=lambda name: len(datasets[name]))
        self.length = int(len(datasets[min_name]) / self.probs[min_name])
        self.seed = seed

    def __len__(self):
        return self.length

    def __iter__(self):
        # pre-calculate language used in each batch
        language_sequence = [0] * self.length
        sample_counts = {key: len(value) for key, value in self.datasets.items()}
        if self.seed is not None:
            random.seed(self.seed)

        for i in range(self.length):
            ratios = [(key, value ** self.factor) for key, value in sample_counts.items() if value > 0]
            assert ratios, "out of samples"
            normalizer = sum(pair[1] for pair in ratios)
            ratios = [(key, value / normalizer) for key, value in ratios]
            
            r = random.random()
            for language, prob in ratios:
                if r < prob:
                    sample_counts[language] -= 1
                    language_sequence[i] = language
                    break
                r -= prob

        # generate batches as determined beforehand
        dataset_iters = {key: iter(value) for key, value in self.datasets.items()}
        for language in language_sequence:
            yield language, next(dataset_iters[language])


class SequentialSampler:
    """Returns all samples in all datasets sequentially,
    mainly used for testing iterations.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = sum(len(value) for value in datasets.values())

    def __len__(self):
        return self.length

    def __iter__(self):
        for name, dataset in self.datasets.items():
            for sample in dataset:
                yield name, sample


class RoundRobinSampler:
    """Converts a dictionary of iterables into an iterable that yields a dictionary.
    Continues yielding dictionary.
    """
    def __init__(self, dict_of_iters, drop_last=False):
        """
        :dict_of_iters, a dictionary of reusable iterables as values
        :drop_last, yield batches until an iterator is used up if True
                else yield batches until all iterators are used up
        """
        self.dict_of_iters = dict_of_iters
        self.drop_last = drop_last
        if self.drop_last:
            self.length = min([len(value) for value in dict_of_iters.values()])
        else:
            self.length = max([len(value) for value in dict_of_iters.values()])

    def __iter__(self):
        iter_dict = {key: iter(value) for key, value in self.dict_of_iters.items()}
        while True:
            next_dict = {}
            for key, value in iter_dict.items():
                try:
                    next_dict[key] = next(value)
                except StopIteration:
                    if self.drop_last:
                        break
                    else:
                        continue

            if not next_dict:
                break

            yield next_dict

    def __len__(self):
        return self.length
