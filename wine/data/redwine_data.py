from wine.data.dataset import Dataset
import pandas as pd
import numpy as np


class RedWine(Dataset):

    def __init__(self, path, subset='train'):
        super(RedWine, self).__init__(path, subset)
        self._read_data()

    def get_data(self):
        """ return attributes, labels """
        data = np.array(self.df)
        return data[:,0:-1], data[:,-1]

    def _read_data(self):
        self.df = pd.read_csv(self.path)
