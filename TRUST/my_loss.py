import  torch
import torch.nn as nn
class Myloss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Myloss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j):  # hi hj是两个不同试图的h : 512 64
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)  # 512*128
        # 转置后 h第i列乘以h.T第i行得到结果就是第i视图中H和第j个视图中同一个实例H的相似性 ，也就是相似矩阵对角线元素
        sim = torch.matmul(h, h.T) / self.temperature_f  # 512*512
        # 第i视图中H和第j个视图中同一个实例H的相似性 ， 其中满足j>i
        sim_i_j = torch.diag(sim, self.batch_size)
        # 第i视图中H和第j个视图中同一个实例H的相似性 ， 其中满足j>i
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        loss /= N
        return loss
