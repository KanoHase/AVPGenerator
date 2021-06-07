import matplotlib.pyplot as plt
import numpy as np
import os
from modlamp.descriptors import GlobalDescriptor


def plot_losses(losses_list, legends_list, file_out):
    assert len(losses_list) == len(legends_list)
    for i, loss in enumerate(losses_list):
        plt.plot(loss, label=legends_list[i])
    plt.legend()
    plt.savefig(file_out)
    plt.close()


def write_samples(sampled_seqs, epoch, best_epoch, g_err_tmp,
                  best_g_err, run_name_dir, a_list, motif_list):
    sample_dir = "./samples/"
    sample_fasta_dir = "./samples_fasta/"
    if not os.path.exists(sample_fasta_dir + run_name_dir):
        os.makedirs(sample_fasta_dir + run_name_dir)
    if not os.path.exists(sample_dir + run_name_dir):
        os.makedirs(sample_dir + run_name_dir)

    with open(sample_dir + run_name_dir + str(epoch) + ".txt", "w") as f:
        for aa_seq in sampled_seqs:
            aa_seq += "\n"
            f.write(aa_seq)
    with open(sample_fasta_dir + run_name_dir + str(epoch) + ".fasta", "w") as f:
        for i, aa_seq in enumerate(sampled_seqs):
            aa_seq += "\n"
            f.write("".join([">", str(i), "\n"]))
            f.write(aa_seq)

    if g_err_tmp/len(sampled_seqs) <= best_g_err:
        best_g_err, best_epoch = write_best_sample(sampled_seqs, epoch, best_epoch, g_err_tmp,
                                                   best_g_err, run_name_dir, a_list, motif_list, sample_dir, sample_fasta_dir)

    return best_g_err, best_epoch


def write_best_sample(sampled_seqs, epoch, best_epoch, g_err_tmp,
                      best_g_err, run_name_dir, a_list, motif_list, sample_dir, sample_fasta_dir):
    print("Samples updated!")
    best_epoch = epoch
    best_g_err = g_err_tmp/len(sampled_seqs)
    best_sample_file = "best_sample"

    with open(sample_dir + run_name_dir + best_sample_file + ".txt", "w") as f:
        for aa_seq in sampled_seqs:
            aa_seq += "\n"
            f.write(aa_seq)
    with open(sample_fasta_dir + run_name_dir + best_sample_file + ".fasta", "w") as f:
        for i, aa_seq in enumerate(sampled_seqs):
            aa_seq += "\n"
            f.write("".join([">", str(i), "\n"]))
            f.write(aa_seq)
    return best_g_err, best_epoch


def makehist_from_intlist(intlist, prop_name, dir_name, run_dir, file_name):
    ave = round(sum(intlist)/len(intlist), 3)
    if not os.path.exists(dir_name + run_dir):
        os.mkdir(dir_name + run_dir)
    with open(dir_name+run_dir+prop_name+"_"+file_name+".txt", "w") as f:
        for num in intlist:
            row = [str(num), "\n"]
            f.write("\t".join(row))
    intarr = np.array(intlist)
    plt.figure()
    plt.hist(intarr)
    plt.savefig(dir_name + run_dir + prop_name + "_" + file_name + ".png")
    return ave


def makehist_from_diclist(diclist, prop_name, dir_name, run_dir, file_name):
    ave_dic = {}
    if not os.path.exists(dir_name + run_dir):
        os.mkdir(dir_name + run_dir)
    for i, dic in enumerate(diclist):
        if i == 0:
            ave_dic = dic
        else:
            for k in dic.keys():
                ave_dic[k] += dic[k]

    with open(dir_name + run_dir + prop_name + "_" + file_name + ".txt", "w") as f:
        for k in dic.keys():
            val = round(ave_dic[k]/len(diclist), 3)
            row = [k, str(val), "\n"]
            ave_dic[k] = val
            f.write("\t".join(row))

    plt.figure()
    x = list(ave_dic.keys())
    y = list(ave_dic.values())
    plt.bar(x, y)
    plt.savefig(dir_name + run_dir + prop_name + "_" + file_name + ".png")
    return ave_dic


def makeintlistdic_from_allep(dir_name, run_dir):
    i = 1
    intlistdic = {}
    len_list, pi_list, hyd_list = [], [], []

    while True:
        if os.path.exists(dir_name+run_dir+str(i)+'.txt'):
            len_sum, pi_sum, hyd_sum = 0, 0, 0
            seq_size = 0

            with open(dir_name+run_dir+str(i)+'.txt') as f:
                for line in f:
                    seq = line[:-1]
                    seq = GlobalDescriptor(seq)
                    seq.length()
                    len_sum += seq.descriptor[0][0]
                    seq.isoelectric_point()
                    pi_sum += seq.descriptor[0][0]
                    seq.hydrophobic_ratio()
                    hyd_sum += seq.descriptor[0][0]
                    seq_size += 1

                len_list.append(round(len_sum/seq_size, 3))
                pi_list.append(round(pi_sum/seq_size, 3))
                hyd_list.append(round(hyd_sum/seq_size, 3))
                i += 1
        else:
            break

    intlistdic["len"] = len_list
    intlistdic["pi"] = pi_list
    intlistdic["hyd"] = hyd_list
    # print(intlistdic, len(len_list))

    return intlistdic


def makeplot_from_intlistdic(intlistdic, eval_dir, run_dir):
    """
    INPUT
    intlistdic: dic(list)
    """
    for k, lis in intlistdic.items():
        with open(eval_dir + run_dir + k + "_allepoch.txt", "w") as f:
            strlis = [str(l) for l in lis]
            f.write("\n".join(strlis))
        plt.figure()
        plt.plot([i for i in range(1, len(lis)+1)], lis)
        plt.savefig(eval_dir + run_dir + k + "_allepoch.png")

    print("DONE: ", run_dir)

    return
