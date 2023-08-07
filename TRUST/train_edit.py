import torch
from network_goal_edit import Network
from metric_edit import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss_edit import Loss
from dataloader_ import load_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

params = {
    "batch_size": 256,
    "temperature_f": 0.7,
    "temperature_l": 0.6,
    "learning_rate": 0.0004,
    "mse_epochs": 200,
    "con_epochs": 50,
    "feature_dim": 512,
    "high_feature_dim": 256,
    "nbrs_num": 4,
    "lamf": 0.2,
    "lamg": 0.3,
    "lamgg": 0.7,

}
# '''
# params = nni.get_next_parameter()
print('params dual con', params)

Dataname = 'Scene'
print("Dataname", Dataname)
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.7)
parser.add_argument("--temperature_l", default=0.6)
parser.add_argument("--learning_rate", default=0.004)
parser.add_argument("--weight_decay", default=0)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200, type=int)  # 200
parser.add_argument("--con_epochs", default=50, type=int)  # 50
parser.add_argument("--tune_epochs", default=50)  # 50
parser.add_argument("--feature_dim", default=512, type=int)
parser.add_argument("--high_feature_dim", default=256, type=int)
parser.add_argument("--nbrs_num", default=4, type=int)
parser.add_argument("--lamf", default=0.2)
parser.add_argument("--lamg", default=0.3)
parser.add_argument("--lamgg", default=0.7)

args = parser.parse_args()

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
'''
if args.dataset == "MNIST-USPS":
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 10

    seed = 10
if args.dataset == "CCV":
    args.con_epochs = 50
    seed = 3
if args.dataset == "Fashion":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-3V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-4V":
    args.con_epochs = 1  # 50
    seed = 10
if args.dataset == "Caltech-5V":
    args.con_epochs = 50
    seed = 5
'''


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


seed = 10

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


def pretrain(epoch, flag_pre):  #
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _, _, _ = model(xs, flag_pre, 10)  # xrs recostruction xv
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch, nbrs_num, flag_con, lam_f, lam_g, lam_gg):  #
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, ss, nbrs_inx = model(xs, flag_con, nbrs_num)

        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(lam_f * criterion.forward_graph(ss[v], ss[w]))
            loss_list.append(lam_g * mes(torch.matmul(hs[v].T, ss[v]), hs[v].T))
            loss_list.append(mes(xs[v], xrs[v]))
        loss_list.append(lam_gg * criterion.forward_neighbor(nbrs_inx, sum(ss) / view, data_size, nbrs_num))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1

for i in range(T):
    print("ROUND:{}".format(i + 1))

    model = Network(data_size, view, dims, args.feature_dim, args.high_feature_dim, class_num, device)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(data_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    flag_con = 1
    flag_pre = 0
    while epoch <= args.mse_epochs:
        pretrain(epoch, flag_pre)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch, args.nbrs_num, flag_con, args.lamf, args.lamg, args.lamgg)
        if epoch == args.mse_epochs + args.con_epochs:
            acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
        epoch += 1
print('canshu', args.lamg)
