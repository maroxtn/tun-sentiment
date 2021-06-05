import random
from torch.utils.data import Sampler

from set_seed import set_seed

set_seed()

class KSampler(Sampler):

    def __init__(self, data_source, batch_size):
        self.lens = [x[1] for x in data_source]
        self.batch_size = batch_size

    def __iter__(self):

        idx = list(range(len(self.lens)))
        arr = list(zip(self.lens, idx))

        random.shuffle(arr)
        n = self.batch_size*100

        iterator = []

        for i in range(0, len(self.lens), n):
            dt = arr[i:i+n]
            dt = sorted(dt, key=lambda x: x[0])

            for j in range(0, len(dt), self.batch_size):
                indices = list(map(lambda x: x[1], dt[j:j+self.batch_size]))
                iterator.append(indices)

        random.shuffle(iterator)
        return iter([item for sublist in iterator for item in sublist])  #Flatten nested list

    def __len__(self):
        return len(self.lens)
