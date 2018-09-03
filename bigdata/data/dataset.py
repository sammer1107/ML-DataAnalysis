"""
This is a base class for wine data set
"""
from abc import ABCMeta
from abc import abstractmethod


class Dataset(metaclass=ABCMeta):
    """ __init__:
        path is the prefix of the file and will be formatted by subset"""
    def __init__(self, path, subset='train'):
        # assert subset in ['train', 'eval'], "subset should be one of ['train', 'eval']"
        self.path = path.format(subset)
        self.subset = subset

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def _read_data(self):
        pass

