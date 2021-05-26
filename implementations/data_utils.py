from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing as sp
import numpy as np
import pandas as pd
import re

import torch
import torchvision.transforms as transforms

data_dir = "./data/"
multi_data_file = "positive_data.txt"
binary_positive_data_file = "positive_540.txt"
binary_negative_data_file = "negative_540.txt"
binary_positive_val_data_file = "val_positive_60.txt"
binary_negative_val_data_file = "val_negative_60.txt"
motif_data_file = "motif.txt"


def load_data(classification, motif, neg=False):  # Use this in WGANgp
    if classification == "binary":
        seq_arr, label_nparr, max_len = prepare_binary(neg=neg)

    if classification == "multi":
        seq_arr, label_nparr, max_len = prepare_multi()

    if motif == True:
        seq_arr, motif_list = motif_restriction(seq_arr)

    else:
        motif_list = []

    seq_nparr, amino_num, a_list = seq_to_onehot(seq_arr, max_len)
    dataset = to_dataloader(seq_nparr, label_nparr)
    # print("==========", seq_nparr.shape, label_nparr.shape, max_len, amino_num, a_list) #(541, 1518) (541,) 46 33 ['D', 'L', 'G', 'P', 'I', '3', 'A', 'K', 'E', 'S', 'H', 'R', 'V', 'T', 'N', 'Q', 'M', '4', 'Y', 'F', 'W', 'C', '6', '2', 'J', '1', '0', '9', 'X', '7', '5', 'O', 'Z']
    return dataset, seq_nparr, label_nparr, max_len, amino_num, a_list, motif_list


def load_data_esm(sampled_seqs=None):
    """
    If pretrain, distinguish pos and neg.
    If not, returns data_esm based on sampled_seqs.

    INPUT
    if pretrain == True
    sampled_seqs: ["MK...G", "KA...E"]

    OUTPUT
    data_esm: [("protein1", "MK...G"), ("protein2", "KA...E")...]
    label_nparr: [0 0 ... 1]
    """
    data_esm = []
    i = 1

    if sampled_seqs != None:
        for seq in sampled_seqs:
            ele = (str(i), seq)
            data_esm.append(ele)
            i += 1
        return data_esm

    else:
        label_list = []  # label (0 or 1)
        pos_dir = data_dir + binary_positive_data_file
        neg_dir = data_dir + binary_negative_data_file
        val_pos_dir = data_dir + binary_positive_val_data_file
        val_neg_dir = data_dir + binary_negative_val_data_file

        dir_list = [pos_dir, neg_dir, val_pos_dir, val_neg_dir]
        for idx, dir in enumerate(dir_list):
            with open(dir) as f:
                for line in f:
                    seq = line[:-1].split()
                    ele = (str(i), seq[0])
                    data_esm.append(ele)
                    i += 1
                    label_list.append(
                        1) if idx % 2 == 0 else label_list.append(0)
            if dir == neg_dir:
                train_size = len(data_esm)

        train_data_esm = data_esm[:train_size]
        val_data_esm = data_esm[train_size:]

        label_nparr = np.array(label_list)
        train_label_nparr = label_nparr[:train_size]
        val_label_nparr = label_nparr[train_size:]

        return train_data_esm, val_data_esm, train_label_nparr, val_label_nparr


def load_data_classify(classification, motif, neg=False):  # Use this in classification
    if classification == "binary":
        seq_arr, train_label_nparr, max_len = prepare_binary(neg=neg)

    if classification == "multi":
        seq_arr, train_label_nparr, max_len = prepare_multi()

    if motif == True:
        seq_arr, motif_list = motif_restriction(seq_arr)

    else:
        motif_list = []

    val_seq_arr, val_label_nparr, val_max_len = prepare_binary(
        pos_dir=data_dir + binary_positive_val_data_file, neg_dir=data_dir + binary_negative_val_data_file, neg=neg)

    if motif == True:
        val_seq_arr, val_motif_list = motif_restriction(val_seq_arr)

    # print(len(seq_arr), len(val_seq_arr), len(seq_arr[10]), len(val_seq_arr[10]))
    train_size = len(seq_arr)

    max_len = max(max_len, val_max_len)
    seq_arr.extend(val_seq_arr)  # concat train seq_arr and test seq_arr

    seq_nparr, amino_num, a_list = seq_to_onehot(
        seq_arr, max_len)  # make all of them one-hot

    train_seq_nparr = seq_nparr[:train_size]  # split
    val_seq_nparr = seq_nparr[train_size:]
    # print("************", seq_nparr.shape, train_seq_nparr.shape, val_seq_nparr.shape)

    return train_seq_nparr, val_seq_nparr, train_label_nparr, val_label_nparr


def prepare_binary(pos_dir=data_dir + binary_positive_data_file, neg_dir=data_dir + binary_negative_data_file, neg=False):
    seq_arr = []  # sequence array(letters by letters) in str
    label_list = []  # label (0 or 1)
    pos_max_len = 0  # max sequence length
    neg_max_len = 0  # max sequence length

    with open(pos_dir) as f:
        for line in f:
            tmp = line[:-1].split()
            seq = list(tmp[0])
            seq_arr.append(seq)
            pre = len(seq)
            pos_max_len = max(pre, pos_max_len)
            label_list.append(1)

    if neg == True:
        with open(neg_dir) as f:
            for line in f:
                tmp = line[:-1].split()
                seq = list(tmp[0])
                seq_arr.append(seq)
                pre = len(seq)
                neg_max_len = max(pre, neg_max_len)
                label_list.append(0)

    label_nparr = np.array(label_list)
    max_len = max(pos_max_len, neg_max_len)

    return seq_arr, label_nparr, max_len


def prepare_multi(which_dir=data_dir + multi_data_file):
    with open(which_dir) as f:
        virus_list = []  # names of all virus
        seq_arr = []  # sequence array(letters by letters) in str
        label_arr_str = []  # label array in str
        max_len = 0  # max sequence length

        heading = f.readline()
        for line in f:  # round1: collect virus data, seq data and label name data
            tmp = line[:-1].split("\t")
            seq = list(tmp[0])
            seq_arr.append(seq)
            pre = len(seq)
            max_len = max(pre, max_len)

            if "," in tmp[1]:
                tmp2 = tmp[1].split(",")

                for v in tmp2:
                    virus_list.append(v)
            else:
                virus_list.append(tmp[1])

            label_arr_str.append(tmp[1])

    virus_list = list(set(virus_list))
    label_nparr = np.zeros((len(seq_arr), len(virus_list)))

    # round2: make multi hot vector from label_arr_str and virus_list
    for i, label in enumerate(label_arr_str):
        if "," in label:
            tmp = label.split(",")

            for v in tmp:
                label_nparr[i, virus_list.index(v)] = 1
        else:
            label_nparr[i, virus_list.index(label)] = 1
    # np.set_printoptions(threshold=np.inf)

    return seq_arr, label_nparr, max_len


def motif_restriction(seq_arr, motif_file=data_dir + motif_data_file, motif_num=10):  # motif_num max:20
    with open(motif_file) as f:
        motif_list = []
        for line in f:
            motif_list.append(line[:-1])

    joined_seq_arr = ["".join(s) for s in seq_arr]

    for i in range(motif_num):  # to search if a part of the sequence matches any motif
        r_motif = re.compile(motif_list[i])
        for j, seq in enumerate(joined_seq_arr):
            trans_seq = re.sub(r_motif, str(i), seq)  # replace motif to index
            joined_seq_arr[j] = trans_seq

    seq_arr = [list(js) for js in joined_seq_arr]
    return seq_arr, motif_list


def seq_to_onehot(seq_arr, max_len):
    a_list = []
    seq_nparr = np.array([])

    for seq in seq_arr:
        for aa in seq:
            if not aa in a_list:
                a_list.append(aa)

    a_list += "Z"

    for i, seq in enumerate(seq_arr):
        zs = max_len - len(seq)
        seq += ["Z"]*zs

        df = pd.DataFrame(seq)

        enc = OneHotEncoder(sparse=False, categories=[a_list])

        seq_oh = enc.fit_transform(df)

        if i == 0:
            seq_nparr = seq_oh

        else:
            seq_nparr = np.block([[[seq_nparr]], [[seq_oh]]])

    seq_nparr = seq_nparr.reshape(-1, len(a_list)*max_len)

    amino_num = len(a_list)

    return seq_nparr, amino_num, a_list


def to_dataloader(seq_nparr, label_nparr):
    dataset = Dataset(seq_nparr, label_nparr)

    return dataset


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, label, transform=True):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            out_data = torch.from_numpy(self.data)[idx]
            out_label = self.label[idx]
        else:
            out_data = self.data[idx]
            out_label = self.label[idx]

        return out_data, out_label
