import matplotlib.pyplot as plt
import numpy as np
import os
from modlamp.descriptors import GlobalDescriptor
import statistics
from sklearn.manifold import TSNE
from implementations.translator import str2tensor


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
    len_ave_list, pi_ave_list, hyd_ave_list, len_var_list, pi_var_list, hyd_var_list = [
    ], [], [], [], [], []

    while True:
        if os.path.exists(dir_name+run_dir+str(i)+'.txt'):
            len_list_ep, pi_list_ep, hyd_list_ep = [], [], []
            seq_size = 0

            with open(dir_name+run_dir+str(i)+'.txt') as f:
                for line in f:
                    seq = line[:-1]
                    seq = GlobalDescriptor(seq)
                    seq.length()
                    len_list_ep.append(seq.descriptor[0][0])
                    seq.isoelectric_point()
                    pi_list_ep.append(seq.descriptor[0][0])
                    seq.hydrophobic_ratio()
                    hyd_list_ep.append(seq.descriptor[0][0])
                    seq_size += 1

                len_ave_list.append(round(len(len_list_ep)/seq_size, 3))
                pi_ave_list.append(round(len(pi_list_ep)/seq_size, 3))
                hyd_ave_list.append(round(len(hyd_list_ep)/seq_size, 3))
                len_var_list.append(
                    round(statistics.pvariance(len_list_ep), 3))
                pi_var_list.append(round(statistics.pvariance(pi_list_ep), 3))
                hyd_var_list.append(
                    round(statistics.pvariance(hyd_list_ep), 3))

                i += 1
        else:
            break

    intlistdic["len_ave"] = len_ave_list
    intlistdic["pi_ave"] = pi_ave_list
    intlistdic["hyd_ave"] = hyd_ave_list
    intlistdic["len_var"] = len_var_list
    intlistdic["pi_var"] = pi_var_list
    intlistdic["hyd_var"] = hyd_var_list
    # print(intlistdic, len(len_ave_list))

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

    return


def makesummary_from_file(eval_dir, run_dirs, var_file_list, last_ep):
    var_dic = {}

    for run_dir in run_dirs:
        run_var_list = []
        header_list = [""]
        for file in var_file_list:
            header_list.append(file)
            last_preds_list = []
            with open(eval_dir + run_dir + "/" + file + ".txt") as f:
                for line in f:
                    last_preds_list.append(float(line[:-1]))
            tmp = last_preds_list[-last_ep:]
            run_var_list.append(sum(tmp)/len(tmp))
        var_dic[run_dir] = run_var_list

    with open(eval_dir + "/var_summary.txt", "w") as f:
        f.write("\t".join(header_list))
        f.write("\n")
        for dir, var_list in var_dic.items():
            tmp = [dir, "\t", str(var_list[0]), "\t", str(
                var_list[1]), "\t", str(var_list[2]), "\n"]
            f.write("".join(tmp))


def maketsne_from_file(file_list, eval_dir):
    a_list = ['D', 'L', 'G', 'P', 'I', 'S', 'E', 'R', 'V', 'T', 'N',
              'A', 'K', 'H', 'Q', 'M', 'Y', 'F', 'W', 'C', 'Z']  # kari
    motif_list = []  # kari
    max_len = 50  # kari

    for i, (dir_name, run_dir, file_name) in enumerate(file_list):
        seq_str = []
        with open(dir_name + run_dir + file_name + ".txt") as f:
            for line in f:
                seq_str.append(list(line[:-1]))

        seq_numpy = str2tensor(seq_str, a_list, motif_list,
                               max_len, output=False)
        #seq_numpy = seq_tens.detach().numpy()

        if i == 0:
            real_seq_numpy = seq_numpy

        else:
            #seq_numpy = seq_numpy[:len(real_seq_numpy)]
            print(seq_numpy.shape, real_seq_numpy.shape)
            print("Creating T-SNE:", run_dir)
            # , "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]
            colors = ["red", "green"]
            plt.figure(figsize=(10, 10))
            points = TSNE(n_components=2,
                          random_state=0).fit_transform(seq_numpy)
            r_points = TSNE(n_components=2, random_state=0).fit_transform(
                real_seq_numpy)

            fig = plt.figure()

            for p in points:
                plt.scatter(p[0], p[1], c=colors[0])
            for r in r_points:
                plt.scatter(r[0], r[1], c=colors[1])

            fig.savefig(eval_dir + run_dir + "tsne.png")
