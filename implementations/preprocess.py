# paper for data augmentation: https://openreview.net/forum?id=Kkw3shxszSd

import argparse
import collections
import os
import random
from re import A
from typing import Sequence
import numpy as np
import Levenshtein
from translator import text2fasta
import pandas as pd
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--fasta",  action='store_true',
                    help="choose whether or not you want to translate txt data to fasta data. Default:False, place --fasta if you want it to be True.")
parser.add_argument("--red",  action='store_true',
                    help="choose whether or not you want to reduce data. Default:False, place --reduce if you want it to be True.")
parser.add_argument("--shuf",  action='store_true',
                    help="choose whether or not you want to augment data with shuffled data. Default:False, place --revr if you want it to be True.")
parser.add_argument("--rep",  action='store_true',
                    help="choose whether or not you want to augment data with replaced amino acid. Default:False, place --rep if you want it to be True.")
parser.add_argument("--revr",  action='store_true',
                    help="choose whether or not you want to augment data with reversed amino acid. Default:False, place --revr if you want it to be True.")
# parser.add_argument("--noscreen",  action='store_true',
#                     help="choose whether or not you want to screen data regarding the type of viruses. Default:False, place --screen if you want it to be True.")
parser.add_argument("--noexp", action='store_true',
                    help="choose if you are using experimented neg data or non-experimented neg data. Default:False(experimented), place --noexp if you want it to be True(non-experimented).")
parser.add_argument("--selectv", action='store_true',
                    help=". Default:False, place --selectv if you want it to be True.")
parser.add_argument("--sim", type=float, default=0.6,
                    help="number of epochs of training")
parser.add_argument("--vir_min", type=int, default=10,
                    help="minimum number of target virus")
parser.add_argument("--max_seqlen", type=int, default=50,
                    help="maximum sequence length")
opt = parser.parse_args()

raw_data_dir = "./raw_data/"
real_pos_file = "positive"
# real_pos_val_file = "val_positive"
real_neg_file = "negative"
real_neg_val_file = "val_negative"
random_file = "random_seq"
opt_dic = {"red": opt.red, "shuf": opt.shuf, "rep": opt.rep, "revr": opt.revr}
filename_end = ""
pos_opt = "posscreen"
neg_opt = "negexpnoexp"
non_a_list = ['B', 'J', 'O', 'U', 'X', 'Z']  # kari
for k, v in opt_dic.items():
    if v == True:
        filename_end += "-" + k
if opt.vir_min == 1:
    pos_opt = "posnoscreen"
if opt.selectv:
    neg_opt = "negexp"
if opt.noexp:
    neg_opt = "negnoexp"

real_data_dir = "./"+"_".join(["real_data", pos_opt, neg_opt]) + "/"
real_fasta_dir = "./"+"_".join(["real_data_fasta", pos_opt, neg_opt]) + "/"
if not os.path.exists(real_data_dir):
    os.mkdir(real_data_dir)
if not os.path.exists(real_fasta_dir):
    os.mkdir(real_fasta_dir)


def reduce(file_list):
    pre = ""
    for dir, p_file in file_list:
        with open(dir + p_file + ".txt") as f:
            with open(dir + p_file + filename_end + ".txt", "w") as g:
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


def select_virus(input_dir, pos_file_list, vir_min):
    """
    Screens target virus that has minimum of vir_min seqs

    INPUT
    pos_file_list: [file names]

    OUTPUT
    dic_of_dic: {file: {seq: vir}}
    v_list: ['HIV', ...]
    """
    pre_v_list = []
    v_list = []
    dic_of_dic = {}

    for file in pos_file_list:
        df = pd.read_csv(input_dir+file, sep="\t")
        seq_vir_dic = dict(zip(df["Sequence"], df["Virus"]))
        dic_of_dic[file] = seq_vir_dic

        for v in seq_vir_dic.values():
            for v_ in v.split(','):
                if v_[0] == ' ':
                    v_ = v_.lstrip()
                if v_[len(v_)-1] == ' ':
                    v_ = v_.rstrip()
                pre_v_list.append(v_)

    v_freq_list = collections.Counter(
        pre_v_list).most_common()  # [('HIV', 279), ...]
    print(v_freq_list)

    if opt.selectv:
        v_list = ["HIV", "HSV", "RSV", "FIV", "MV", "Influenza", "SARS-CoV"]
    else:
        for virus, num in v_freq_list:
            if num >= vir_min:
                v_list.append(virus)
            else:
                break
    return v_list, dic_of_dic


def screen_pos(output_dir, dic_of_dic, pos_file_list, v_list):
    """
    Counts screend pos data size and make screened pos data files

    INPUT
    pos_file_list: [file names]
    dic_of_dic: {file: {seq: vir}}
    v_list: ['HIV', ...]

    OUTPUT
    seqnum_dic: {'positive.txt': 469, ...}
    """
    print("#############Screening positive file#############")

    seqnum_dic = {}

    for file in pos_file_list:
        seqnum_dic[file] = 0
        with open(output_dir+file, "w") as f:
            pos_dic = dic_of_dic[file]
            for seq, vir in pos_dic.items():
                virlis = vir.split(',')
                invalid_flag = False  # True if there are invalid letters
                for v in virlis:
                    if v in v_list:  # check if there are frequent viruses
                        for na in non_a_list:  # check if there are invalid letters
                            if na in seq:
                                print("Contains invalid letters:", seq)
                                invalid_flag = True
                                break
                        if len(seq) > opt.max_seqlen:  # check if seq has less than max_seqlen length
                            print("Too long: ", seq)
                            invalid_flag = True
                        if invalid_flag == False:
                            seqnum_dic[file] += 1
                            f.write("".join([seq, "\n"]))
                            break
    return seqnum_dic


def cut_invalid_seqs(seq, non_a_list):
    invalid_flag = False
    for na in non_a_list:
        if (na in seq) or (len(seq) > opt.max_seqlen):
            print("Invalid: ", seq)
            invalid_flag = True
            break
    return invalid_flag


def write_random_nonexp(f, adding_num, input_file):
    # replacing exp file name to non-exp file name
    nonexp_file = input_file.replace('_exp.txt', '_noexp.txt')
    df = pd.read_csv(nonexp_file, sep="\t")
    seq_list = df['Sequence'].tolist()

    random.seed(1)
    random.shuffle(seq_list)

    for seq in seq_list[:adding_num]:
        f.write("".join([seq, '\n']))


def screen_neg(input_dir, output_dir, neg_file_list, seqnum_dic, noexp):
    """
    Make screened neg data files regarding pos data size
    """
    print("#############Screening negative file#############")
    if noexp:  # Default: False
        neg_input_file = [file for file in neg_file_list if "noexp" in file]
    else:  # Default: Here
        neg_input_file = [
            file for file in neg_file_list if "noexp" not in file]

    for k, pos_num in seqnum_dic.items():
        if "val" in k:
            # only one of the files will be chosen
            target_file = [file for file in neg_input_file if "val" in file]
            out_negfile = real_neg_val_file + ".txt"
        else:
            # only one of the files will be chosen
            target_file = [
                file for file in neg_input_file if "val" not in file]
            out_negfile = real_neg_file + ".txt"

        df = pd.read_csv(input_dir+target_file[0], sep="\t")
        seq_list = df['Sequence'].tolist()

        with open(output_dir+out_negfile, 'w') as f:
            neg_valid_num = 0
            for seq in seq_list:
                invalid_flag = cut_invalid_seqs(seq, non_a_list)
                if not invalid_flag and neg_valid_num < pos_num:
                    neg_valid_num += 1
                    f.write("".join([seq, "\n"]))
                elif neg_valid_num >= pos_num:
                    print("Reduced Negative data size to:", pos_num)
                    break

            if neg_valid_num < pos_num:
                print("The size of {}({}) is less than {}({})".format(
                    target_file[0], neg_valid_num, k, pos_num))
                if not opt.noexp:
                    write_random_nonexp(f, pos_num-neg_valid_num,
                                        raw_data_dir+target_file[0])
                    print("Added", pos_num-neg_valid_num,
                          "non-experimented data")


def main():
    input_dir = raw_data_dir
    output_dir = real_data_dir
    input_file_list = os.listdir(input_dir)
    print(input_file_list)

    pos_file_list = []
    neg_file_list = []
    for file in input_file_list:
        if "pos" in file:
            pos_file_list.append(file)
        if "neg" in file:
            neg_file_list.append(file)
    v_list, dic_of_dic = select_virus(
        input_dir, pos_file_list, opt.vir_min)
    seqnum_dic = screen_pos(output_dir, dic_of_dic,
                            pos_file_list, v_list)
    screen_neg(input_dir, output_dir, neg_file_list, seqnum_dic, opt.noexp)

    if opt.red:
        reduce([[real_data_dir, real_pos_file]])
    if opt.shuf or opt.rep:
        augment_data([[real_data_dir, real_pos_file]],
                     opt.red, opt.shuf, opt.rep, opt.revr)

    real_file_list_tmp = os.listdir(real_data_dir)
    real_file_list = [[real_data_dir, file] for file in real_file_list_tmp]
    text2fasta(real_file_list, real_fasta_dir)

    shutil.copyfile(raw_data_dir+random_file+'.txt',
                    real_data_dir+random_file+'.txt')

    print('Your option is: ', "_".join([pos_opt, neg_opt]))


if __name__ == '__main__':
    main()
