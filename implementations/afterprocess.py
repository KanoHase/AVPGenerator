import matplotlib.pyplot as plt
import numpy as np
import re
import rstr
import os

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

        aa_samples = tensor_to_str(g_fake_data, a_list, motif)

        with open(sample_dir + sample_file, "w") as f:
            for i, aa_seq in enumerate(aa_samples):
                f.write("".join([">", str(i), "\n"]))
                f.write(aa_seq)

    return best_g_err, best_epoch   


def tensor_to_str(g_fake_data, a_list, motif_list):
    aa_samples = []
    g_fake_data = g_fake_data.to('cpu').detach().numpy().copy()

    for seq in g_fake_data:
        seq_str = ""

        for aa in seq:
            aa_idx = np.argmax(aa)
            which_amino = a_list[aa_idx]
            if re.match(r'[0-9]+', which_amino) and motif_list: #if motiif was used
                motif_tmp = rstr.xeger(motif_list[int(which_amino)]) #to generate random sequence regarding the motif
                # print(motif_tmp, motif_list[int(which_amino)])
                seq_str += motif_tmp

            elif which_amino !="Z":
                seq_str += which_amino
        
        seq_str += "\n"
        aa_samples.append(seq_str)

    return aa_samples
