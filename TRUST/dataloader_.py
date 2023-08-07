from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path)
       # print('data',data)
        scaler = MinMaxScaler()

        self.view1 = scaler.fit_transform(data['X'][0][0].astype(np.float32).transpose())
        self.view2 = scaler.fit_transform(data['X'][0][1].astype(np.float32).transpose())
        self.view3 = scaler.fit_transform(data['X'][0][2].astype(np.float32).transpose())

        self.labels = scipy.io.loadmat(path)['gt']



    def __len__(self):
        return 4485

    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
           self.view2[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()





def load_data(dataset):
    if dataset == "Scene":
        dataset = BDGP('./data/scene15.mat')
        dims = [20, 59, 40]
        view = 3
        data_size = 4485
        class_num = 15
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
