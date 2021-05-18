import argparse
import os
import glob
import numpy as np

from implementations.data_utils import prepare_binary, load_data_classify
from implementations.afterprocess import plot_losses, write_samples
from esm_master.initialize_esm import gen_repr
from models import *

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--show_loss", type=int, default=5,
                    help="number of epochs of showing loss")
parser.add_argument("--show_test_result", type=int,
                    default=100, help="number of epochs of showing loss")
parser.add_argument("--figure_dir", type=str,
                    default="./figures/", help="directory name to put figures")
parser.add_argument("--classification", type=str, default="binary",
                    help="binary or multi for discriminator classification task")
parser.add_argument("--which_model", type=str,
                    default="Dis_Lin_classify", help="choose network model")

opt = parser.parse_args()


class TransClassifier():
    def __init__(self, in_dim, out_dim, hidden, batch):
        self.batch = batch
        self.hidden = hidden
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.classification = opt.classification
        self.which_model = opt.which_model
        self.figure_dir = opt.figure_dir
        self.use_cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.checkpoint_dir = "./checkpoint/classification/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.build_model()

    def build_model(self):
        if self.which_model == "Dis_Lin_classify":
            self.model = Dis_Lin_classify(
                self.in_dim, self.out_dim, self.hidden)
        else:
            self.model = []  # temporary
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(
            self.checkpoint_dir + "weights.pth"))

    def prepare_data_esm(self, batch_seqs):
        data_esm = []
        i = 1
        for seq in batch_seqs:
            ele = (str(i), seq)
            data_esm.append(ele)
            i += 1
        return data_esm

    def analyse_function(self, seq_repr):
        data = Variable(self.Tensor(seq_repr))
        all_preds_posneg = self.model(data)
        all_preds = all_preds_posneg[:, 1]
        all_preds = all_preds.to('cpu').detach().numpy().copy()
        print("Made predictions...")
        return all_preds


"""
def main():
    use_cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    seq_arr, label_nparr, max_len = prepare_binary(neg=True)
    sampled_seqs = []
    for seq in seq_arr:
        sampled_seqs.append("".join(seq))

    # in_dim, out_dim, hidden, batch
    FA = TransClassifier(768, 2, 512, 16)

    data_esm = FA.prepare_data_esm(sampled_seqs)
    seq_repr = gen_repr(data_esm)

    all_preds = FA.analyse_function(seq_repr, 16)
    prediction = all_preds.data.max(1)[1]  # indices of max
    target = Variable(Tensor(label_nparr))  # .reshape(-1, 1))
    target = target.to(dtype=torch.long)

    prediction_sum = prediction.eq(target.data).sum(
    ) if use_cuda else prediction.eq(target.data).sum().numpy()  # 正解率
    accuracy = prediction_sum / len(label_nparr)

    print("ACCURACY: ", accuracy)


if __name__ == '__main__':
    main()
"""
