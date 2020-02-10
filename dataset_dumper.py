"""
    Utility for dumping a dataset to disk (to enable efficient multi-threaded runs)
    Date created: 8/27/19
    Python Version: 3.6
"""

__author__ = "Guy Gaziv"

from GPUtils.startup_guyga import *
from torch.utils.data import Dataset

class DumpedDataset(Dataset):
    def __init__(self, wrapped_dataset, folder_path=None):
        self.wrapped_dataset = wrapped_dataset
        if folder_path is None:
            self.folder_path = './tmp/'
        else:
            self.folder_path = folder_path
        overridefolder(self.folder_path)
        for index, item in enumerate(self.wrapped_dataset):
            with open(pjoin(self.folder_path, str(index)), 'wb') as f:
                pickle.dump(item, f)

    def __getitem__(self, index):
        with open(pjoin(self.folder_path, str(index)), 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.wrapped_dataset)

    def __getattr__(self, attr):
        return getattr(self, attr) if attr in self.__dict__ else getattr(self.wrapped_dataset, attr)


if __name__ == '__main__':
    pass
