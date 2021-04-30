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
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    sample_file = "samples.txt"

    if g_err_tmp/len(g_fake_data) < best_g_err:
        best_epoch = epoch
        best_g_err = g_err_tmp/len(g_fake_data)

        aa_samples = tensor2str(g_fake_data, a_list, motif)

        with open(sample_dir + sample_file, "w") as f:
            for i, aa_seq in enumerate(aa_samples):
                f.write("".join([">", str(i), "\n"]))
                f.write(aa_seq)

    return best_g_err, best_epoch   
