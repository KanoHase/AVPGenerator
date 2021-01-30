from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing as sp
import numpy as np
import pandas as pd
import os
import re

import torch
import torchvision
import torchvision.transforms as transforms

data_dir = "./data/"
multi_data_file = "positive_data.txt"
binary_positive_data_file = "amino_positive_541.txt"
binary_negative_data_file = "amino_negative_541.txt"
motif_data_file = "motif.txt"


def load_data(classification, motif, neg=False):
    if classification == "binary":
        seq_arr, label_nparr, max_len = prepare_binary(neg = neg)

    if classification == "multi":
        seq_arr, label_nparr, max_len = prepare_multi()

    if motif == True:
        seq_arr, motif_list = motif_restriction(seq_arr)
    
    else:
        motif_list = []

    seq_nparr, amino_num, a_list = seq_to_onehot(seq_arr, max_len)
    dataset = to_dataloader(seq_nparr, label_nparr)

    print(seq_nparr.shape, label_nparr.shape, max_len, amino_num, a_list)
    return dataset, seq_nparr, label_nparr, max_len, amino_num, a_list, motif_list


def prepare_binary(pos_dir = data_dir + binary_positive_data_file, neg_dir = data_dir + binary_negative_data_file, neg = False):
    seq_arr = [] #sequence array(letters by letters) in str
    label_list = [] #label (0 or 1)
    pos_max_len = 0 #max sequence length
    neg_max_len = 0 #max sequence length

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

def prepare_multi(which_dir = data_dir + multi_data_file):
    with open(which_dir) as f:
        virus_list = [] #names of all virus
        seq_arr = [] #sequence array(letters by letters) in str
        label_arr_str = [] #label array in str
        max_len = 0 #max sequence length

        heading = f.readline()
        for line in f: #round1: collect virus data, seq data and label name data
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
    label_nparr = np.zeros((len(seq_arr),len(virus_list)))

    for i, label in enumerate(label_arr_str): #round2: make multi hot vector from label_arr_str and virus_list
        if "," in label:
            tmp = label.split(",")

            for v in tmp:
                label_nparr[i, virus_list.index(v)] = 1
        else:
            label_nparr[i, virus_list.index(label)] = 1
    #np.set_printoptions(threshold=np.inf)

    return seq_arr, label_nparr, max_len


def motif_restriction(seq_arr, motif_file = data_dir + motif_data_file, motif_num=10): #motif_num max:20
    with open (motif_file) as f:
        motif_list = []
        for line in f:
            motif_list.append(line[:-1])
    
    joined_seq_arr = ["".join(s) for s in seq_arr]
    
    for i in range(motif_num): #to search if a part of the sequence matches any motif
        r_motif = re.compile(motif_list[i])
        for j, seq in enumerate(joined_seq_arr):
            trans_seq = re.sub(r_motif, str(i), seq) #replace motif to index
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

        enc = OneHotEncoder( sparse=False, categories = [a_list] )

        seq_oh = enc.fit_transform(df)

        if i == 0 :
            seq_nparr = seq_oh
        
        else:
            seq_nparr = np.block([[[seq_nparr]], [[seq_oh]]])
    
    seq_nparr = seq_nparr.reshape(-1, len(a_list)*max_len)

    amino_num = len(a_list)
    
    return seq_nparr, amino_num, a_list


def to_dataloader(seq_nparr, label_nparr):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Dataset(seq_nparr, label_nparr, transform)

    return dataset


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
          out_data = self.transform(self.data)[0][idx]
          out_label = self.label[idx]
        else:
          out_data = self.data[idx]
          out_label =  self.label[idx]

        return out_data, out_label

#load_data("multi")
#load_data("binary", True)

#dataset = Dataset(data, label, transform)