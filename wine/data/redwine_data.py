from wine.data.dataset import Dataset
import pandas as pd


class RedWine(Dataset):

    train_path = 'data/winequality-red-preprocessed-train.csv'
    eval_path = 'data/winequality-red-preprocessed-eval.csv'

    def __init__(self, subset='train'):
        super(RedWine, self).__init__(subset)
        self._read_data()

    def get_data(self):
        return self.df

    def _read_data(self):
        path = self.train_path if self.subset == 'train' else self.eval_path
        self.df = pd.read_csv(path)