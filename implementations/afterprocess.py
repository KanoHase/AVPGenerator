import matplotlib.pyplot as plt
import numpy as np
import re
import rstr
import os
from implementations.translator import tensor2str


def plot_losses(losses_list, legends_list, file_out):
    assert len(losses_list) == len(legends_list)
    for i, loss in enumerate(losses_list):
        plt.plot(loss, label=legends_list[i])
    plt.legend()
    plt.savefig(file_out)
    plt.close()


def write_samples(g_err_tmp, best_g_err, epoch, best_epoch, g_fake_data, a_list, motif):
    sample_dir = "./samples/"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    sample_file = "samples.txt"
    sample_fasta_file = "samples.fasta"

    if g_err_tmp/len(g_fake_data) < best_g_err:
        best_epoch = epoch
        best_g_err = g_err_tmp/len(g_fake_data)

        aa_samples = tensor2str(g_fake_data, a_list, motif)

        with open(sample_dir + sample_file, "w") as f:
            for aa_seq in aa_samples:
                f.write(aa_seq)
        with open(sample_dir + sample_fasta_file, "w") as f:
            for i, aa_seq in enumerate(aa_samples):
                f.write("".join([">", str(i), "\n"]))
                f.write(aa_seq)

    return best_g_err, best_epoch


def makehist_from_intlist(intlist, prop_name, dir_name, file_name):
    ave = round(sum(intlist)/len(intlist), 3)
    with open(dir_name+prop_name+"_"+file_name+".txt", "w") as f:
        for num in intlist:
            row = [str(num), "\n"]
            f.write("\t".join(row))
    intarr = np.array(intlist)
    plt.figure()
    plt.hist(intarr)
    plt.savefig(dir_name + prop_name + "_" + file_name + ".png")
    return ave


def makehist_from_diclist(diclist, prop_name, dir_name, file_name):
    ave_dic = {}
    for i, dic in enumerate(diclist):
        if i == 0:
            ave_dic = dic
        else:
            for k in dic.keys():
                ave_dic[k] += dic[k]

    with open(dir_name + prop_name + "_" + file_name + ".txt", "w") as f:
        for k in dic.keys():
            val = round(ave_dic[k]/len(diclist), 3)
            row = [k, str(val), "\n"]
            ave_dic[k] = val
            f.write("\t".join(row))
    # print(list(ave_dic.keys()), list(ave_dic.values()))
    plt.figure()
    x = list(ave_dic.keys())
    y = list(ave_dic.values())
    # plt.figure(figsize=(5,5))
    plt.bar(x, y)
    plt.savefig(dir_name + prop_name + "_" + file_name + ".png")
    return ave_dic
