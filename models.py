import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from implementations.torch_utils import *


class Gen_Lin(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super(Gen_Lin, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 100)
        self.fc3 = nn.Linear(100, in_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Dis_Lin_classify(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super(Dis_Lin_classify, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden, 100)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class Gen_Lin_Block(nn.Module):  # DEFAULT
    def __init__(self, in_dim, out_dim, hidden):
        super(Gen_Lin_Block, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, in_dim),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        return x


class Dis_Lin(nn.Module):  # DEFAULT
    def __init__(self, in_dim, out_dim, hidden):
        super(Dis_Lin, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, out_dim),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class Gen_Lin_Block_CNN(nn.Module):
    def __init__(self, in_dim, max_len, amino_num, hidden, batch):
        super(Gen_Lin_Block_CNN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden*max_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, amino_num, 1)
        self.max_len = max_len
        self.hidden = hidden
        self.batch = batch

    def forward(self, noise):
        output = self.fc1(noise)
        # (BATCH_SIZE, DIM, SEQ_LEN)
        output = output.view(-1, self.hidden, self.max_len)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(self.batch*self.max_len, -1)
        output = gumbel_softmax(output, 0.5)
        return output.view(shape)  # (BATCH_SIZE, SEQ_LEN, len(charmap))


class Dis_Lin_Block_CNN(nn.Module):
    def __init__(self, in_dim, max_len, amino_num, hidden):
        super(Dis_Lin_Block_CNN, self).__init__()
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1d = nn.Conv1d(amino_num, hidden, 1)
        self.linear = nn.Linear(max_len*hidden, 1)
        self.max_len = max_len
        self.hidden = hidden

    def forward(self, input):
        output = input.transpose(1, 2)  # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.max_len*self.hidden)
        output = self.linear(output)
        return output


class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),  # nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),  # nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)


class Generator_FBGAN(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Generator_FBGAN, self).__init__()
        self.fc1 = nn.Linear(128, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden

    def forward(self, noise):
        output = self.fc1(noise)
        # (BATCH_SIZE, DIM, SEQ_LEN)
        output = output.view(-1, self.hidden, self.seq_len)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len, -1)
        output = gumbel_softmax(output, 0.5)
        return output.view(shape)  # (BATCH_SIZE, SEQ_LEN, len(charmap))


class Discriminator_FBGAN(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Discriminator_FBGAN, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)
        self.linear = nn.Linear(seq_len*hidden, 1)

    def forward(self, input):
        output = input.transpose(1, 2)  # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len*self.hidden)
        output = self.linear(output)
        return output
