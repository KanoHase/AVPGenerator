import argparse
import os

from implementations.data_utils import load_data
from implementations.afterprocess import plot_losses, write_samples
from implementations.torch_utils import to_var, calc_gradient_penalty
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
parser.add_argument("--d_steps", type=int, default=10, help="number of epochs to train generator")
parser.add_argument("--lr", type=int, default=0.0001, help="learning rate")
parser.add_argument("--figure_dir", type=str, default="./figures/", help="directory name to put figures")
parser.add_argument("--classification", type=str, default="binary", help="binary or multi for discriminator classification task")
parser.add_argument("--generator_model", type=str, default="Gen_Lin_Block", help="choose generator model")
parser.add_argument("--discriminator_model", type=str, default="Dis_Lin", help="choose discriminator model")
parser.add_argument("--loss", type=str, default="WGAN-gp", help="choose loss")
parser.add_argument("--optimizer", type=str, default="Adam", help="choose optimizer")
parser.add_argument("--motif", type=bool, default=True, help="choose whether or not you want to include motif restriction")

opt = parser.parse_args()
classification = opt.classification 
generator_model = opt.generator_model
discriminator_model = opt.discriminator_model
optimizer = opt.optimizer
figure_dir = opt.figure_dir
if not os.path.exists(figure_dir): os.makedirs(figure_dir)
use_cuda  = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# def prepare_data_model():
dataset, seq_nparr, label_nparr, max_len, amino_num, a_list, motif_list = load_data(classification, opt.motif) #numpy.ndarray
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
print(dataloader, seq_nparr, max_len, amino_num)

if classification == "multi":
    out_dim = len(y[0])
if classification == "binary":
    out_dim = 2

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

# print(G)
# print(D)

if optimizer == "Adam":
    G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.9))


def train_model():
    d_fake_losses, d_real_losses, grad_penalties = [],[],[]
    G_losses, D_losses, W_dist = [],[],[]

    one = torch.tensor(1, dtype=torch.float)
    one = one.cuda() if use_cuda else one
    one_neg = one * -1

    best_g_err = np.inf
    best_epoch = 0

    for epoch in range(opt.epoch):
        g_err_tmp = 0
        g_fake_data_all = []

        for i, (data, _) in enumerate(dataloader):
            real_data = Variable(data.type(Tensor))   

            D.zero_grad()
            
            z_input = Variable(Tensor(np.random.normal(0, 1, (opt.batch, max_len*amino_num))))
            d_fake_data = G(z_input)
            d_real_pred = D(real_data)
            d_fake_pred = D(d_fake_data)

            gradient_penalty = calc_gradient_penalty(real_data.data, d_fake_data.data, opt.batch, D)
            d_err = -torch.mean(d_real_pred) + torch.mean(d_fake_pred) + gradient_penalty
            
            d_err.backward()
            D_optimizer.step()
            G.zero_grad()

            if i % opt.d_steps == 0: # Train D
                g_fake_data = G(z_input)
                dg_fake_pred = D(g_fake_data)
                g_err = -torch.mean(dg_fake_pred)
                g_err_tmp += g_err #to calculate average later, and to see if the value was lower than the last value

                g_err.backward()
                G_optimizer.step()

                if g_fake_data_all == []:
                    g_fake_data_all = g_fake_data
                
                else:
                    g_fake_data_all = torch.cat((g_fake_data_all,g_fake_data),0)
                

            # Append things for logging
            d_fake_np, d_real_np, gp_np = (torch.mean(d_fake_pred).data).cpu().numpy(), \
                    (torch.mean(d_real_pred).data).cpu().numpy(), (gradient_penalty.data).cpu().numpy()
            grad_penalties.append(gp_np)
            d_real_losses.append(d_real_np)
            d_fake_losses.append(d_fake_np)
            D_losses.append(d_fake_np - d_real_np + gp_np) # minus(real - fake)
            G_losses.append((g_err.data).cpu().numpy())
            W_dist.append(d_real_np - d_fake_np)

            if i % opt.show_loss == 0:
                summary_str = 'Iteration {} - loss_d: {}, loss_g: {}, w_dist: {}, grad_penalty: {}'\
                    .format(i, (d_err.data).cpu().numpy(),
                    (g_err.data).cpu().numpy(), ((torch.mean(d_real_pred) - torch.mean(d_fake_pred)).data).cpu().numpy(), gp_np)
                # print(summary_str)
                plot_losses([G_losses, D_losses], ["gen", "disc"], figure_dir + "losses.png")
                plot_losses([W_dist], ["w_dist"], figure_dir + "dist.png")
                plot_losses([grad_penalties],["grad_penalties"], figure_dir + "grad.png")
                plot_losses([d_fake_losses, d_real_losses],["d_fake", "d_real"], figure_dir + "d_loss_components.png")

        #write amino acid data when g_err is low            
        if epoch > 1:
            g_fake_data_all = g_fake_data_all.reshape(-1, max_len, amino_num)
            best_g_err, best_epoch = write_samples(g_err_tmp, best_g_err, epoch, best_epoch, g_fake_data_all, a_list, motif_list)

    print('Best epoch:{}, Minimum g_error:{}'.format(best_epoch, best_g_err))


def main():
    train_model()

if __name__ == '__main__':
    main()
