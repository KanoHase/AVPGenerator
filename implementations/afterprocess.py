import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from implementations.translator import str2tensor
import seaborn as sns


def make_plot(losses_list, legends_list, file_out):
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
