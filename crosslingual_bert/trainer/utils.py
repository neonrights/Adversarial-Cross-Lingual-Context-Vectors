import random


class SmoothedRandomSampler:
    """Randomly sample from datasets in proportion to their size downsampled
    by a factor.
    """
    def __init__(self, datasets, factor=0.7):
        self.datasets = datasets
        probs = {key: len(value)**factor for key, value in datasets.items()}
        total = sum(probs.values())
        self.probs = {name: prob / total for name, prob in probs.items()}
        min_name = min(datasets, key=lambda name: len(datasets[name]))
        self.length = int(len(datasets[min_name]) / self.probs[min_name])

    def __len__(self):
        return self.length

    def __iter__(self):
        datasets = list(self.datasets)
        iter_data = {key: iter(value) for key, value in self.datasets.items()}
        for _ in range(self.length):
            sample = None
            while sample is None:
                assert datasets, "out of samples"
                name, = random.choices(datasets,
                        weights=[self.probs[name] for name in datasets])
                try:
                    sample = next(iter_data[name])
                    yield name, sample
                except StopIteration:
                    # remove empty dataset from candidates
                    datasets.remove(name)


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
