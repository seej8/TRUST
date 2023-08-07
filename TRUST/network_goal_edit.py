import torch.nn as nn
from torch.nn.functional import normalize
import torch
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


'''
class Tensor(nn.Module):
    def __init__(self, batch_size):
        super(Tensor, self).__init__()
        self.tensor = nn.Parameter(torch.Tensor(batch_size, batch_size), requires_grad=True)
       # self.tensor.data.fill_(0)

    def forward(self):
        return self.tensor
'''


class Tensor(nn.Module):
    def __init__(self, batch_size, view):
        super(Tensor, self).__init__()
        self.tensors = []
        for v in range(view):
            self.tensors.append(nn.Parameter(torch.Tensor(batch_size, batch_size), requires_grad=True))

    def forward(self, h, view):
        hs_dot_ss = []
        for v in range(view):
            h_dot_s = torch.matmul(h[v].T, self.tensors[v])
            hs_dot_ss.append(h_dot_s)

        return hs_dot_ss


class Network(nn.Module):
    def __init__(self, batch_size, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.batch_size = batch_size

        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        # self.hs, self.tensors = Tensor(batch_size, view)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),

        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, batch_size)
            # nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs, flag, nbrs_num):
        hs = []
        qs = []
        xrs = []
        zs = []
        ss = []
        nbrs_v = np.zeros((self.batch_size, nbrs_num))
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            s = self.label_contrastive_module(z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
            ss.append(s)
        if flag == 1:
            # 10 neighbors
            h_all = sum(hs) / self.view
            X_nb = h_all.cpu().detach().numpy()
            nbrs = NearestNeighbors(n_neighbors=nbrs_num + 1, algorithm='auto').fit(X_nb)
            dis, idx = nbrs.kneighbors(X_nb)
            for i in range(self.batch_size):
                for j in range(nbrs_num):
                    nbrs_v[i][j] += idx[i][j]
            nbrs_v = np.array(nbrs_v).astype(int)

        return hs, qs, xrs, zs, ss, nbrs_v

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds
