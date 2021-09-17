from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import pandas as pd

raw_dir = "./raw_data/"
real_pos_file = "val_positive"
real_pos_train_file = "positive"
sample_dir = "./samples_posscreen_negexpnoexp/utP-PS_fe0.75_fp0.5_mp0.0_revNone/"
eval_dir = "./eval_posscreen_negexpnoexp/utP-PS_fe0.75_fp0.5_mp0.0_revNone/"
file_name = "99.txt"
result_file = "align_result.txt"
align_file = "alignment.txt"


def make_seq_list(path):
    seq_list = []

    with open(path) as f:
        for line in f:
            seq = line[:-1]
            seq_list.append(seq)
    return seq_list


def pairwise(seq_list, train_path, val_path, result_path, align_path):
    '''
    make pairwise alignment with real data
    '''
    train_df = pd.read_csv(train_path, sep="\t")
    val_df = pd.read_csv(val_path, sep="\t")
    train_seq_v_dic = dict(zip(train_df['Sequence'], train_df['Virus']))
    val_seq_v_dic = dict(zip(val_df['Sequence'], val_df['Virus']))
    realseq_list = list(train_seq_v_dic.keys())
    realseq_list += list(val_seq_v_dic.keys())

    with open(result_path, 'w') as f:
        with open(align_path, 'w') as g:
            f.write('\t'.join(['Sequence', 'Sequence with a highest score',
                               'Alignment Score', 'Possible target virus', 'train or val', '\n']))
            for synseq in seq_list:
                max_score = 0
                sim_seq = ''
                best_alignment = None

                for realseq in realseq_list:
                    if len(realseq) > 50:  # seq with length over 50 iis excluded
                        continue
                    alignments = pairwise2.align.globalxx(
                        synseq, realseq)
                    if max(max_score, alignments[0][2]) == alignments[0][2]:
                        max_score = alignments[0][2]  # aligment score
                        sim_seq = realseq
                        best_alignment = alignments[0]

                print(format_alignment(*best_alignment))

                tv = 'train' if sim_seq in train_seq_v_dic else 'val'
                sim_virus = train_seq_v_dic[sim_seq] if tv == 'train' else val_seq_v_dic[sim_seq]

                f.write(
                    '\t'.join([synseq, sim_seq, str(max_score), sim_virus, tv, '\n']))
                g.write(''.join([format_alignment(*best_alignment), '\n']))


def main():
    seq_list = make_seq_list(sample_dir+file_name)
    pairwise(seq_list, raw_dir+real_pos_train_file +
             ".txt", raw_dir+real_pos_file+".txt", eval_dir+result_file, eval_dir+align_file)


if __name__ == '__main__':
    main()
