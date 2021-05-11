import argparse

from implementations.data_utils import load_data_classify, load_data_esm
from implementations.afterprocess import plot_losses, write_samples
from esm_master.initialize_esm import gen_repr
from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=10000,
                    help="number of epochs of training")
parser.add_argument("--hidden", type=int, default=512,
                    help="number of neurons in hidden layer")
parser.add_argument("--batch", type=int, default=64,
                    help="number of batch size")
parser.add_argument("--show_loss", type=int, default=5,
                    help="number of epochs of showing loss")
parser.add_argument("--show_test_result", type=int,
                    default=100, help="number of epochs of showing loss")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")
parser.add_argument("--figure_dir", type=str,
                    default="./figures/", help="directory name to put figures")
parser.add_argument("--classification", type=str, default="binary",
                    help="binary or multi for discriminator classification task")
parser.add_argument("--discriminator_model", type=str,
                    default="Dis_Lin_classify", help="choose discriminator model")
parser.add_argument("--loss", type=str, default="WGAN-gp", help="choose loss")
parser.add_argument("--optimizer", type=str,
                    default="SGD", help="choose optimizer")
parser.add_argument("--motif", action='store_true',
                    help="choose whether or not you want to include motif restriction. Default:False, place --motif if you want it to be True.")
parser.add_argument("--transformer", action='store_true',
                    help="choose whether or not you want to use transformer representation. Default:False, place --transformer if you want it to be True. WARNING: motif restriction cannot be used if transformer == True")

opt = parser.parse_args()
classification = opt.classification
discriminator_model = opt.discriminator_model
optimizer = opt.optimizer
figure_dir = opt.figure_dir
use_cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

if opt.transformer:
    train_data_esm, val_data_esm, train_label_nparr, val_label_nparr = load_data_esm()
    # data_esm: [('1079', 'ALVGATFGCGVPTI')...]
    train_seq_repr = gen_repr(train_data_esm)
    val_seq_repr = gen_repr(val_data_esm)
    val_X = val_seq_repr
    val_y = val_label_nparr

    train_dataloader = torch.utils.data.DataLoader(
        train_seq_repr, batch_size=opt.batch, shuffle=True, drop_last=True)

else:
    train_dataset, train_label_nparr, val_seq_nparr, val_label_nparr, max_len, amino_num = load_data_classify(
        classification, opt.motif, neg=True)  # numpy.ndarray

    val_X = val_seq_nparr
    val_y = val_label_nparr

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
    #print(vars(train_dataset), seq_nparr, seq_nparr.shape, label_nparr, label_nparr.shape, max_len, amino_num)

if classification == "multi":
    out_dim = len(train_label_nparr[0])
if classification == "binary":
    out_dim = 2

pre_accuracy = 0

torch.manual_seed(1)  # seed固定、ネットワーク定義前にする必要ありそう

if discriminator_model == "Dis_Lin_classify":
    model = Dis_Lin_classify(max_len, amino_num, out_dim, opt.hidden)

if optimizer == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=0.02)

train_loss = []
train_accu = []

if use_cuda:
    model = model.cuda()

for epoch in range(1, opt.epoch):
    model.train()  # 学習モード

    for i, (X, y) in enumerate(train_dataloader):
        data = Variable(X.type(Tensor))  # 微分可能な型
        target = Variable(y.type(Tensor))  # .reshape(-1, 1))
        target = target.to(dtype=torch.long)
        optimizer.zero_grad()  # 勾配初期化
        output = model(data)  # データを流す

        loss = F.nll_loss(output, target)
        loss.backward()  # バックプロパゲーション

        train_loss.append(loss.data.item())
        optimizer.step()   # 重み更新

        prediction = output.data.max(1)[1]  # indices of max
        prediction_sum = prediction.eq(target.data).sum(
        ) if use_cuda else prediction.eq(target.data).sum().numpy()  # 正解率
        accuracy = prediction_sum / len(data)
        train_accu.append(accuracy)

    if epoch % opt.show_loss == 0:
        print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(
            int(epoch/opt.show_loss), loss.data.item(), sum(train_accu)/len(train_accu)))

    if epoch % opt.show_test_result == 0:
        model.eval()  # 推論モード
        target_v = Variable(Tensor(val_y))  # .reshape(-1, 1))
        output_v = model(Variable(Tensor(val_X)))
        _, prediction_v = torch.max(output_v.data, 1)
        prediction_v_sum = prediction_v.eq(target_v.data).sum(
        ) if use_cuda else prediction_v.eq(target_v.data).sum().numpy()  # 正解率
        accuracy_v = prediction_v_sum / len(prediction_v)

        if accuracy_v > pre_accuracy:
            print(model)
            pre_accuracy = accuracy_v

        print('Test Step: {}\tAccuracy: {:.3f}'.format(
            int(epoch/opt.show_test_result), accuracy_v))