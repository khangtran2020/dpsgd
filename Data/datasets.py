import torch
import numpy as np
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, X, y, ismale):
        self.ismale = torch.from_numpy(ismale)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.ismale[index]

    def __len__(self):
        return self.len

