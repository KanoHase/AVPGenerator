# paper for data augmentation: https://openreview.net/forum?id=Kkw3shxszSd

import argparse
import os
import random
import numpy as np
import Levenshtein
from translator import text2fasta

parser = argparse.ArgumentParser()
parser.add_argument("--red",  action='store_true',
                    help="choose whether or not you want to reduce data. Default:False, place --reduce if you want it to be True.")
parser.add_argument("--shuf",  action='store_true',
                    help="choose whether or not you want to augment data with shuffled data. Default:False, place --revr if you want it to be True.")
parser.add_argument("--rep",  action='store_true',
                    help="choose whether or not you want to augment data with replaced amino acid. Default:False, place --rep if you want it to be True.")
parser.add_argument("--revr",  action='store_true',
                    help="choose whether or not you want to augment data with reversed amino acid. Default:False, place --revr if you want it to be True.")
parser.add_argument("--sim", type=float, default=0.6,
                    help="number of epochs of training")
opt = parser.parse_args()

data_dir = "./data/"
fasta_dir = "./fasta_data/"
real_pos_file = "positive"
real_pos_val_file = "val_positive"
real_neg_file = "negative_noexp"
real_neg_val_file = "val_negative_noexp"
opt_dic = {"red": opt.red, "shuf": opt.shuf, "rep": opt.rep, "revr": opt.rev}
filename_end = ""
for k, v in opt_dic.items():
    if v == True:
        filename_end += "_" + k


def reduce(file_list):
    pre = ""
    for dir, p_file in file_list:
        with open(dir + p_file + ".txt") as f:
            with open(data_dir + p_file + filename_end + ".txt", "w") as g:
                for line in f:
                    seq = line[:-1]
                    similarity = Levenshtein.distance(pre, seq)
                    divider = len(pre) if len(pre) > len(seq) else len(seq)
                    similarity = similarity / divider
                    similarity = 1 - similarity
                    if similarity <= opt.sim:
                        g.write(line[:-1])
                        g.write("\n")
                    pre = seq


def augment_data(file_list, red, shuf, rep, revr):
    seq_lis = []
    rep_dic = {'A': 'V', 'V': 'A', 'S': 'T', 'T': 'S', 'F': 'Y', 'Y': 'F', 'K': 'R', 'R': 'K',
               'C': 'M', 'M': 'C', 'D': 'E', 'E': 'D', 'N': 'Q', 'Q': 'N', 'V': 'I', 'I': 'V'}
    dir = file_list[0][0]
    p_file = file_list[0][1]

    if red:
        with open(dir + p_file + filename_end + ".txt") as f:
            for line in f:
                seq_lis.append(line[:-1])

    else:
        with open(dir + p_file + ".txt") as f:
            for line in f:
                seq_lis.append(line[:-1])

    if rep:
        total_len = sum(len(seq) for seq in seq_lis)
        flag_lis = [False, True]
        prob_lis = [0.99, 0.01]
        rep_lis = np.random.choice(
            a=flag_lis, size=total_len, p=prob_lis)

    with open(dir + p_file + filename_end + ".txt", "w") as f:
        i = 0
        rep_flag = False  # if the seq has replacement

        for seq in seq_lis:
            row = [seq, "\n"]
            repseq = ""  # if the seq has replacement, this will be added to the dataset

            if shuf:
                start = random.randint(0, len(seq)-2)
                end = random.randint(start+2, len(seq))
                shufpart = seq[start:end]  # shuffle this part
                shufpart = ''.join(random.sample(shufpart, len(shufpart)))
                shufseq = seq[0:start] + shufpart + seq[end:len(seq)] + "\n"
                row.append(shufseq)

            for aa in seq:
                if rep and rep_lis[i]:
                    rep_flag = True
                    if aa in rep_dic:
                        repseq += rep_dic[aa]  # replace amino
                    else:
                        repseq += aa
                else:
                    repseq += aa
                i += 1

            if rep_flag:
                repseq = repseq + "\n"
                row.append(repseq)
                rep_flag = False

            if revr:
                revseq = seq[::-1]
                revseq = revseq + "\n"
                row.append(revseq)
            f.write("".join(row))


def main():
    file_list = [[data_dir, real_pos_file]]
    other_file_list = [[data_dir, real_neg_file], [
        data_dir, real_pos_val_file], [data_dir, real_neg_val_file]]

    text2fasta(file_list, fasta_dir)
    text2fasta(other_file_list, fasta_dir)

    if opt.red:
        reduce(file_list)
    if opt.shuf or opt.rep:
        augment_data(file_list, opt.red, opt.shuf, opt.rep, opt.revr)


if __name__ == '__main__':
    main()
