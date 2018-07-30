"""
This is a base class for wine data set
"""
from abc import ABCMeta
from abc import abstractmethod


class Dataset(metaclass=ABCMeta):

    def __init__(self, subset='train'):
        assert subset in ['train', 'eval'], "subset should be one of ['train', 'eval']"
        self.subset = subset

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def _read_data(self):
        pass

