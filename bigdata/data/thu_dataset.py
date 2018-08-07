from bigdata.data.dataset import Dataset
import os
import numpy as np
import pandas as pd


class ThuDataset(Dataset):
    """class for Thu Big Data Competition dataset
    path : this argument should be the directory of the csv files"""

    def __init__(self, path, subset='train'):
        super().__init__(path, subset)
        self._read_data()

    def get_data(self):
        return self.attrs, self.targets

    def _read_data(self):
        self.attrs = []
        self.targets = []

        for csv in os.listdir(self.path):
            df = pd.read_csv(os.path.join(self.path, csv), header=0, index_col=0)
            self.attrs.append(np.array(df.iloc[0:-1]))
            self.targets.append(np.float32(df.iloc[-1,0]))

        self.attrs = np.array(self.attrs)
        self.targets = np.array(self.targets)
