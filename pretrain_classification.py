import argparse
import os

from implementations.data_utils import load_data_classify, load_data_esm, to_dataloader
from implementations.afterprocess import plot_losses, write_samples
from esm_master.initialize_esm import gen_repr
from models import *

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=2000,
                    help="number of epochs of training")
parser.add_argument("--hidden", type=int, default=512,
                    help="number of neurons in hidden layer")
parser.add_argument("--batch", type=int, default=64,
                    help="number of batch size")
parser.add_argument("--show_loss", type=int, default=5,
                    help="number of epochs of showing loss")
parser.add_argument("--show_test_result", type=int,
                    default=100, help="number of epochs of showing loss")
parser.add_argument("--lr", type=int, default=0.02, help="learning rate")
parser.add_argument("--figure_dir", type=str,
                    default="./figures/", help="directory name to put figures")
parser.add_argument("--classification", type=str, default="binary",
                    help="binary or multi for discriminator classification task")
parser.add_argument("--classifier_model", type=str,
                    default="Dis_Lin_classify", help="choose discriminator model")
parser.add_argument("--optimizer", type=str,
                    default="SGD", help="choose optimizer")
parser.add_argument("--motif", action='store_true',
                    help="choose whether or not you want to include motif restriction. Default:False, place --motif if you want it to be True. WARNING: cannot be used with notransformer option")
parser.add_argument("--revise",  action='store_true',
                    help="choose whether or not you want to revise data. Default:False, place --revise if you want it to be True.")
parser.add_argument("--notransformer", action='store_false',
                    help="choose whether or not you want to use transformer representation. Default:True, place --notransformer if you do not want transformer.")

opt = parser.parse_args()
classification = opt.classification
classifier_model = opt.classifier_model
transformer = opt.notransformer
figure_dir = opt.figure_dir
use_cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
checkpoint_dir = "./checkpoint/classification/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def load_data_pretrain(transformer):
    if transformer:
        train_data_esm, val_data_esm, train_label_nparr, val_label_nparr = load_data_esm(
            revise=opt.revise)
        # data_esm: [('1079', 'ALVGATFGCGVPTI')...]
        train_seq_nparr = gen_repr(train_data_esm)
        val_seq_repr = gen_repr(val_data_esm)

        val_X = val_seq_repr
        val_y = val_label_nparr

        train_dataset = to_dataloader(train_seq_nparr, train_label_nparr)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch, shuffle=True, drop_last=True)

        in_dim = len(train_seq_nparr[0])

    else:
        train_seq_nparr, val_seq_nparr, train_label_nparr, val_label_nparr = load_data_classify(
            classification, opt.motif, revise=opt.revise)  # numpy.ndarray

        val_X = val_seq_nparr
        val_y = val_label_nparr

        train_dataset = to_dataloader(train_seq_nparr, train_label_nparr)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
        #print(vars(train_dataset), seq_nparr, seq_nparr.shape, label_nparr, label_nparr.shape, max_len, amino_num)
        in_dim = len(train_seq_nparr[0])

    return train_dataloader, train_seq_nparr, train_label_nparr, val_X, val_y, in_dim


def train_model():
    optimizer = opt.optimizer
    train_dataloader, train_seq_nparr, train_label_nparr, val_X, val_y, in_dim = load_data_pretrain(
        transformer)

    if classification == "multi":
        out_dim = len(train_label_nparr[0])
    if classification == "binary":
        out_dim = 2

    pre_accuracy = 0

    torch.manual_seed(1)  # seed固定、ネットワーク定義前にする必要ありそう

    if classifier_model == "Dis_Lin_classify":
        model = Dis_Lin_classify(in_dim, out_dim, opt.hidden)

    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=opt.lr)

    train_loss = []
    train_accu = []

    if use_cuda:
        model = model.cuda()

    for epoch in range(1, opt.epoch):
        model.train()  # 学習モード

        for _, (X, y) in enumerate(train_dataloader):
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

            if accuracy_v >= pre_accuracy:
                torch.save(model.state_dict(), checkpoint_dir +
                           "weights.pth")
                print("MODEL SAVED")
                pre_accuracy = accuracy_v

            print('Test Step: {}\tAccuracy: {:.3f}'.format(
                int(epoch/opt.show_test_result), accuracy_v))


def main():
    train_model()


if __name__ == '__main__':
    main()
