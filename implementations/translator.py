import numpy as np
import re
import rstr

from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def text2fasta(filename_list):
    fasta_dir = "fasta_data"

    for p_file in filename_list:
        with open("./data/" + p_file + ".txt") as f:
            with open("./" + fasta_dir + "/" + p_file + ".fasta", "w") as g:
                i = 1
                for line in f:
                    tmp = ">" + str(i) + "\n" + line[:-1] + "\n"
                    g.write(tmp)
                    i += 1


def tensor2str(seq_tensor, a_list, motif_list, output=True):
    aa_samples = []
    seq_nparr = seq_tensor.to('cpu').detach().numpy().copy()

    for seq in seq_nparr:
        seq_str = ""

        for aa in seq:
            aa_idx = np.argmax(aa)
            which_amino = a_list[aa_idx]
            if re.match(r'[0-9]+', which_amino) and motif_list:  # if motif was used
                # to generate random sequence regarding the motif
                motif_tmp = rstr.xeger(motif_list[int(which_amino)])
                # print(motif_tmp, motif_list[int(which_amino)])
                seq_str += motif_tmp

            elif which_amino != "Z":
                seq_str += which_amino
        if output:
            seq_str += "\n"
        aa_samples.append(seq_str)

    return aa_samples

# text2fasta(["positive_540", "negative_540"])


def str2tensor(seq_str, a_list, motif_list, max_len, output=False):
    seq_nparr = []
    for i, seq in enumerate(seq_str):
        zs = max_len - len(seq)
        seq += ["Z"]*int(zs)

        df = pd.DataFrame(seq)

        enc = OneHotEncoder(sparse=False, categories=[a_list])

        seq_oh = enc.fit_transform(df)

        if i == 0:
            seq_nparr = seq_oh

        else:
            seq_nparr = np.block([[[seq_nparr]], [[seq_oh]]])

    seq_nparr = seq_nparr.reshape(-1, len(a_list)*max_len)

    return seq_nparr
