import numpy as np
import re
import rstr
import os

from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def text2fasta(file_list, fasta_dir):
    '''
    INPUT
    file_list:list(list(dir, file_name))
    fasta_dir = str
    '''
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)

    for dir, p_file in file_list:
        input_path = dir + p_file if ".txt" in p_file else dir + p_file + ".txt"
        with open(input_path) as f:
            output_path = fasta_dir + \
                p_file.replace(
                    '.txt', '.fasta') if ".txt" in p_file else fasta_dir + p_file + ".fasta"
            with open(output_path, "w") as g:
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


def list2fasta(seq_list, out_path):
    with open(out_path, "w") as f:
        i = 1
        for seq in seq_list:
            tmp = ">" + str(i) + "\n" + seq + "\n"
            f.write(tmp)
            i += 1


def fasta2txt(fasta_path, txt_path):
    with open(fasta_path) as f:
        with open(txt_path, 'w') as g:
            for line in f:
                if '>' not in line:
                    g.write(line)
