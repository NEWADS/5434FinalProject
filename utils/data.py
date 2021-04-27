import torch
import logging
import numpy as np
import sklearn

from torch.utils.data import Dataset


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    x = logging.getLogger(__name__)
    x.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    x.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    x.addHandler(console)
    return x


def set_seed(digit: int):
    assert digit, 'Please specify a non-zero number when calling this func.'
    torch.manual_seed(digit)
    np.random.seed(digit)


class SequenceDataset(Dataset):
    def __init__(self, x: str, shuffle: bool = True, seed: int = 1, balanced=True, delta_1=1, delta_2=1):
        super(SequenceDataset, self).__init__()
        raw_data = np.load(x)
        self.features = raw_data['reads'].astype(np.float32)
        try:
            self.labels = raw_data['label']
        except KeyError:
            self.labels = np.ones(self.features.shape[0]) * -1
        self.labels = self.labels.astype(np.int64)
        self.ids = np.arange(len(self.labels)).astype(np.int64)
        if shuffle:
            self.features, self.labels, self.ids = sklearn.utils.shuffle(self.features,
                                                                         self.labels,
                                                                         self.ids, random_state=seed)
        if balanced:
            x_train_0, y_train_0 = self.features[np.where(self.labels == 0)[0]], self.labels[np.where(self.labels == 0)[0]]
            ids_0 = self.ids[np.where(self.labels == 0)[0]]
            x_train_1, y_train_1 = self.features[np.where(self.labels == 1)[0]], self.labels[np.where(self.labels == 1)[0]]
            ids_1 = self.ids[np.where(self.labels == 1)[0]]
            # randomly sample some data to make the dataset looks more equally distributed.
            x_train_1, y_train_1, ids_1 = x_train_1[:int(19713 * delta_1), :], y_train_1[:int(19713 * delta_1)], ids_1[:int(19713 * delta_1)]
            x_train_2, y_train_2 = self.features[np.where(self.labels == 2)[0]], self.labels[np.where(self.labels == 2)[0]]
            ids_2 = self.ids[np.where(self.labels == 2)[0]]
            # randomly sample some data to make the dataset looks more equally distributed.
            x_train_2, y_train_2, ids_2 = x_train_2[:int(19713 * delta_2), :], y_train_2[:int(19713 * delta_2)], ids_2[:int(19713 * delta_2)]
            self.features = np.concatenate([x_train_0, x_train_1, x_train_2], axis=0)
            self.labels = np.concatenate([y_train_0, y_train_1, y_train_2], axis=0)
            self.ids = np.concatenate([ids_0, ids_1, ids_2], axis=0)
            self.features, self.labels, self.ids = sklearn.utils.shuffle(self.features,
                                                                         self.labels,
                                                                         self.ids, random_state=seed)
        self.seed = seed

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature, label, index = self.features[idx], self.labels[idx], self.ids[idx]
        # feature = [_encode(i) for i in feature]
        return {'feature': torch.tensor(feature, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.int64),
                'index': torch.tensor(index, dtype=torch.int64)}
