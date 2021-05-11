import argparse
import os
import math

from implementations.data_utils import load_data, update_data
from implementations.afterprocess import plot_losses, write_samples
from implementations.torch_utils import to_var, calc_gradient_penalty
from implementations.fb_utils import meta_select_pos
from implementations.translator import tensor2str, str2tensor
from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500,
                    help="number of epochs of training")
parser.add_argument("--hidden", type=int, default=512,
                    help="number of neurons in hidden layer")
parser.add_argument("--batch", type=int, default=16,
                    help="number of batch size")
parser.add_argument("--show_loss", type=int, default=5,
                    help="number of epochs of showing loss")
parser.add_argument("--d_steps", type=int, default=10,
                    help="number of epochs to train generator")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--preds_cutoff", type=float,
                    default=0.8, help="threshold of preds")
parser.add_argument("--figure_dir", type=str,
                    default="./figures/", help="directory name to put figures")
parser.add_argument("--classification", type=str, default="binary",
                    help="binary or multi for discriminator classification task")
parser.add_argument("--fbtype", type=str, default="MetaiAVP",
                    help="Transformer or MetaiAVP for feedback task")
parser.add_argument("--generator_model", type=str,
                    default="Gen_Lin_Block", help="choose generator model")
parser.add_argument("--discriminator_model", type=str,
                    default="Dis_Lin", help="choose discriminator model")
parser.add_argument("--loss", type=str, default="WGAN-gp", help="choose loss")
parser.add_argument("--optimizer", type=str,
                    default="Adam", help="choose optimizer")
parser.add_argument("--motif", action='store_true',
                    help="choose whether or not you want to include motif restriction. Default:False, place --motif if you want it to be True.")
parser.add_argument("--nofb", action='store_true',
                    help="choose whether or not you want to feedback data. Default:False, place --nofb if you do not need FB.")

opt = parser.parse_args()
classification = opt.classification
fbtype = opt.fbtype
generator_model = opt.generator_model
discriminator_model = opt.discriminator_model
optimizer = opt.optimizer
figure_dir = opt.figure_dir
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
use_cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
out_dim = 2


def train_model():
    dataset, seq_nparr, label_nparr, max_len, amino_num, a_list, motif_list = load_data(
        classification, opt.motif)  # numpy.ndarray
    # if nofb == True, dataset and seq_nparr must be random amino, not AVP
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
    order_label = np.zeros(len(seq_nparr))
    # (541, 1472) 46 32 #seq_nparr=(label_nparr.shape, max_len*amino_num)
    # print("!!!!!!!!", seq_nparr.shape, max_len,
    #       amino_num, len(label_nparr), a_list)

    G, D = prepare_model(max_len, amino_num)

    if optimizer == "Adam":
        G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    d_fake_losses, d_real_losses, grad_penalties = [], [], []
    G_losses, D_losses, W_dist = [], [], []

    one = torch.tensor(1, dtype=torch.float)
    one = one.cuda() if use_cuda else one

    best_g_err = np.inf
    best_epoch = 0

    for epoch in range(1, opt.epoch):
        g_err_tmp = 0
        g_fake_data_all = []

        if opt.nofb == False:  # if you're using FeedBack
            # we need sampled_seq here
            # sampled_seq:['MDR...','ACD...'...]
            sample_itr = math.floor(len(seq_nparr)/opt.batch)  # kari
            sampled_seqs = generate_sample(
                sample_itr, opt.batch, max_len, amino_num, G, a_list, motif_list)

            if fbtype == "MetaiAVP":  # if you're using MetaiAVP as a Function Analyser
                pos_seqs = meta_select_pos(
                    sampled_seqs, epoch, opt.preds_cutoff)
                pos_nparr = str2tensor(
                    pos_seqs, a_list, motif_list, max_len, output=False)
                # print("!!!!!!!!!!!", pos_nparr, pos_nparr.shape)

            dataset, seq_nparr, order_label = update_data(
                pos_nparr, seq_nparr, order_label, label_nparr, epoch)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch, shuffle=False, drop_last=True)

        for i, (data, _) in enumerate(dataloader):
            real_data = Variable(data.type(Tensor))
            D.zero_grad()

            z_input = Variable(Tensor(np.random.normal(
                0, 1, (opt.batch, max_len*amino_num))))  # (64, 1518)
            d_fake_data = G(z_input)
            d_real_pred = D(real_data)
            d_fake_pred = D(d_fake_data)

            gradient_penalty = calc_gradient_penalty(
                real_data.data, d_fake_data.data, opt.batch, D)
            d_err = -torch.mean(d_real_pred) + \
                torch.mean(d_fake_pred) + gradient_penalty

            d_err.backward()
            D_optimizer.step()
            G.zero_grad()

            if i % opt.d_steps == 0:  # Train D
                g_fake_data = G(z_input)
                dg_fake_pred = D(g_fake_data)
                g_err = -torch.mean(dg_fake_pred)
                # to calculate average later, and to see if the value was lower than the last value
                g_err_tmp += g_err

                g_err.backward()
                G_optimizer.step()

                if g_fake_data_all == []:
                    g_fake_data_all = g_fake_data

                else:
                    g_fake_data_all = torch.cat(
                        (g_fake_data_all, g_fake_data), 0)

            # Append things for logging
            d_fake_np, d_real_np, gp_np = (torch.mean(d_fake_pred).data).cpu().numpy(), \
                (torch.mean(d_real_pred).data).cpu().numpy(
            ), (gradient_penalty.data).cpu().numpy()
            grad_penalties.append(gp_np)
            d_real_losses.append(d_real_np)
            d_fake_losses.append(d_fake_np)
            D_losses.append(d_fake_np - d_real_np +
                            gp_np)  # minus(real - fake)
            G_losses.append((g_err.data).cpu().numpy())
            W_dist.append(d_real_np - d_fake_np)

            if i % opt.show_loss == 0:
                summary_str = 'Iteration {} - loss_d: {}, loss_g: {}, w_dist: {}, grad_penalty: {}'\
                    .format(i, (d_err.data).cpu().numpy(),
                            (g_err.data).cpu().numpy(), ((torch.mean(d_real_pred) - torch.mean(d_fake_pred)).data).cpu().numpy(), gp_np)
                print(summary_str)
                plot_losses([G_losses, D_losses], ["gen", "disc"],
                            figure_dir + "losses.png")
                plot_losses([W_dist], ["w_dist"], figure_dir + "dist.png")
                plot_losses([grad_penalties], ["grad_penalties"],
                            figure_dir + "grad.png")
                plot_losses([d_fake_losses, d_real_losses], [
                            "d_fake", "d_real"], figure_dir + "d_loss_components.png")

        # write amino acid data when g_err is low
        if epoch > 1:
            g_fake_data_all = g_fake_data_all.reshape(-1, max_len, amino_num)
            best_g_err, best_epoch = write_samples(
                g_err_tmp, best_g_err, epoch, best_epoch, g_fake_data_all, a_list, motif_list)

    print('Best epoch:{}, Minimum g_error:{}'.format(best_epoch, best_g_err))


def prepare_model(max_len, amino_num):
    # if classification == "multi":
    #     out_dim = len(y[0])

    if generator_model == "Gen_Lin_Block_CNN":
        G = Gen_Lin_Block_CNN(max_len, amino_num, out_dim, opt.hidden)
    if discriminator_model == "Dis_Lin_Block_CNN":
        D = Dis_Lin_Block_CNN(max_len, amino_num, out_dim, opt.hidden)
    if generator_model == "Gen_Lin_Block":
        G = Gen_Lin_Block(max_len, amino_num, out_dim, opt.hidden)
    if discriminator_model == "Dis_Lin":
        D = Dis_Lin(max_len, amino_num, out_dim, opt.hidden)

    if use_cuda:
        G = G.cuda()
        D = D.cuda()

    print(G)  # cuda が合ってもなくても変わらない結果だった
    print(D)

    return G, D


def generate_sample(sample_itr, batch_size, max_len, amino_num, G, a_list, motif_list):
    sampled_seqs = []
    for _ in range(sample_itr):
        z = to_var(torch.randn(batch_size, max_len*amino_num))
        G.eval()
        seqs = G(z)
        seqs = seqs.reshape(-1, max_len, amino_num)
        sampled_seqs += tensor2str(seqs, a_list, motif_list, output=False)
    G.train()
    return sampled_seqs


def main():
    train_model()


if __name__ == '__main__':
    main()
