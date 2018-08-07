from bigdata.data.dataset import Dataset

class ThuDataset(Dataset):

    def __init__(self, path, subset='train'):
        super().__init__(path, subset)

    def get_data(self):
        pass

    def _read_data(self):
        pass
