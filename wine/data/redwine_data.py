from wine.data.dataset import Dataset
import pandas as pd
import numpy as np


class RedWine(Dataset):

    train_path = 'wine/data/winequality-red-preprocessed-train.csv'
    eval_path = 'wine/data/winequality-red-preprocessed-eval.csv'

    def __init__(self, subset='train'):
        super(RedWine, self).__init__(subset)
        self._read_data()

    def get_data(self):
        """ return attributes, labels """
        data = np.array(self.df)
        return data[:,0:-1], data[:,-1]

    def _read_data(self):
        path = self.train_path if self.subset == 'train' else self.eval_path
        self.df = pd.read_csv(path)