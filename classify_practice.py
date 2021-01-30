import argparse

from implementations.data_utils import load_data
from implementations.afterprocess import plot_losses, write_samples
from implementations.torch_utils import to_var, calc_gradient_penalty
from sklearn.model_selection import train_test_split
from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500, help="number of epochs of training")
parser.add_argument("--hidden", type=int, default=512, help="number of neurons in hidden layer")
parser.add_argument("--batch", type=int, default=64, help="number of batch size")
parser.add_argument("--show_loss", type=int, default=5, help="number of epochs of showing loss")
parser.add_argument("--show_test_result", type=int, default=100, help="number of epochs of showing loss")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")
parser.add_argument("--figure_dir", type=str, default="./figures/", help="directory name to put figures")
parser.add_argument("--classification", type=str, default="binary", help="binary or multi for discriminator classification task")
parser.add_argument("--discriminator_model", type=str, default="Dis_Lin_classify", help="choose discriminator model")
parser.add_argument("--loss", type=str, default="WGAN-gp", help="choose loss")
parser.add_argument("--optimizer", type=str, default="SGD", help="choose optimizer")
parser.add_argument("--motif", type=bool, default=True, help="choose whether or not you want to include motif restriction")

opt = parser.parse_args()
classification = opt.classification 
discriminator_model = opt.discriminator_model
optimizer = opt.optimizer
figure_dir = opt.figure_dir
use_cuda  = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# def prepare_data_model():
dataset, seq_nparr, label_nparr, max_len, amino_num, a_list, motif_list = load_data(classification, opt.motif, neg=True) #numpy.ndarray
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
print(dataloader, seq_nparr, seq_nparr.shape, label_nparr, label_nparr.shape, max_len, amino_num)

if classification == "multi":
    out_dim = len(y[0])
if classification == "binary":
    out_dim = 2

X = seq_nparr
y = label_nparr

if classification == "multi":
    out_dim = len(y[0])
if classification == "binary":
    out_dim = 2

#学習データと検証データを分割
train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size = 0.2, random_state=71)

# tensor型に変換
train_X = torch.Tensor(train_X)
val_X = torch.Tensor(val_X)
train_y = torch.LongTensor(train_y)
val_y = torch.LongTensor(val_y)

torch.manual_seed(71) #seed固定、ネットワーク定義前にする必要ありそう

if discriminator_model == "Dis_Lin_classify":
    model = Dis_Lin_classify(max_len, amino_num, out_dim, opt.hidden)

if optimizer == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=0.02)

train_loss = []
train_accu = []
i = 1
j = 1

for epoch in range(opt.epoch):
    model.train() #学習モード
    data = Variable(train_X)#微分可能な型
    target =  Variable(train_y)#.reshape(-1, 1))
    optimizer.zero_grad() #勾配初期化
    output = model(data) #データを流す
    # print("=========", output.shape, target.shape)

    # loss = nn.BCELoss() #loss計算
    # loss(output, target)
    #loss = nn.CrossEntropyLoss() #loss計算
    #out = loss(output, target)
    loss = F.nll_loss(output, target)
    loss.backward()    #バックプロパゲーション
    # out.backward()    #バックプロパゲーション

    train_loss.append(loss.data.item())
    optimizer.step()   # 重み更新
        
    prediction = output.data.max(1)[1] #予測結果
    accuracy = prediction.eq(target.data).sum().numpy() / len(train_X) #正解率
    train_accu.append(accuracy)
    
    if i % opt.show_loss == 0:
        print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data.item(), accuracy))
    
    if i % opt.show_test_result == 0:
        model.eval() #推論モード
        outputs = model(Variable(val_X))
        _, predicted = torch.max(outputs.data, 1)
        print('Test Step: {}\tAccuracy: {:.3f}'.format(j, predicted.eq(val_y).sum().numpy() / len(predicted)))
        j += 1

    i += 1
    
#print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data.item(), accuracy))
