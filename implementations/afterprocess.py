import matplotlib.pyplot as plt
import numpy as np
import os
from implementations.translator import tensor2str


def plot_losses(losses_list, legends_list, file_out):
    assert len(losses_list) == len(legends_list)
    for i, loss in enumerate(losses_list):
        plt.plot(loss, label=legends_list[i])
    plt.legend()
    plt.savefig(file_out)
    plt.close()


def write_samples(g_fake_data_all, epoch, best_epoch, g_err_tmp,
                  best_g_err, run_name_dir, a_list, motif_list):
    sample_dir = "./samples/"
    # if not os.path.exists(sample_dir):
    #     os.makedirs(sample_dir)
    sample_fasta_dir = "./samples_fasta/"
    if not os.path.exists(sample_fasta_dir + run_name_dir):
        os.makedirs(sample_fasta_dir + run_name_dir)
    if not os.path.exists(sample_dir + run_name_dir):
        os.makedirs(sample_dir + run_name_dir)

    aa_samples = tensor2str(g_fake_data_all, a_list, motif_list)

    with open(sample_dir + run_name_dir + str(epoch) + ".txt", "w") as f:
        for aa_seq in aa_samples:
            f.write(aa_seq)
    with open(sample_fasta_dir + run_name_dir + str(epoch) + ".fasta", "w") as f:
        for i, aa_seq in enumerate(aa_samples):
            f.write("".join([">", str(i), "\n"]))
            f.write(aa_seq)

    if g_err_tmp/len(g_fake_data_all) <= best_g_err:
        best_g_err, best_epoch = write_best_sample(g_fake_data_all, epoch, best_epoch, g_err_tmp,
                                                   best_g_err, run_name_dir, a_list, motif_list, sample_dir, sample_fasta_dir)

    return best_g_err, best_epoch


def write_best_sample(g_fake_data_all, epoch, best_epoch, g_err_tmp,
                      best_g_err, run_name_dir, a_list, motif_list, sample_dir, sample_fasta_dir):
    print("Samples updated!")
    best_epoch = epoch
    best_g_err = g_err_tmp/len(g_fake_data_all)
    best_sample_file = "best_sample"

    aa_samples = tensor2str(g_fake_data_all, a_list, motif_list)

    with open(sample_dir + run_name_dir + best_sample_file + ".txt", "w") as f:
        for aa_seq in aa_samples:
            f.write(aa_seq)
    with open(sample_fasta_dir + run_name_dir + best_sample_file + ".fasta", "w") as f:
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
