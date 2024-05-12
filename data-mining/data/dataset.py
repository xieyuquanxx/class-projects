import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def get_data_label(df, train: bool = True):
    label = None
    if train:
        data = df.iloc[:, :-7].values
        label = df.iloc[:, -7:].values
    else:
        data = df.values
    return torch.from_numpy(data), torch.from_numpy(label) if label is not None else None


class SteelPlateDataset(Dataset):
    def __init__(self, data_path: str, train: bool = True):
        super().__init__()
        self._data = pd.read_csv(data_path)
        self._x, self._y = get_data_label(self._data, train)

        self.train = train

    def __getitem__(self, index):
        if not self.train:
            return self._x[index].to(dtype=torch.float32), {}
        assert self._y is not None
        return self._x[index].to(dtype=torch.float32), self._y[index].to(dtype=torch.float32)

    def __len__(self):
        return self._x.shape[0]


if __name__ == "__main__":
    dataset = SteelPlateDataset("/data/Users/xyq/developer/data-mining/src/data/train_data.csv")
    print(dataset[1])
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for _, data in enumerate(loader):
        print(data)
        break
