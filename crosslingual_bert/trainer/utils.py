class IterDict:
    """Converts a dictionary of iterables into an iterable that yields a dictionary.
    Continues yielding dictionary 
    """
    def __init__(self, dict_of_iters, drop_last=False):
        self.dict_of_iters = dict_of_iters
        if drop_last:
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
                    if drop_last:
                        break
                    else:
                        continue

            if not next_dict:
                break

            yield next_dict

    def __len__(self):
        return self.length
