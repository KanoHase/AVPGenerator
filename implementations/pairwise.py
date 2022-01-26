from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import pandas as pd
import re
import math

column_idx = ['Synthetic Sequence', 'Real Sequence with a Highest Score',
              'Alignment Score', 'Target Virus', 'Train or Val']
tvposv_dic = {'HIV': 279, 'HSV': 61, 'RSV': 57, 'FIV': 51, 'MV': 49, 'HCV': 45, 'HPF3': 37, 'WNV': 30,
              'paramyxoviruses': 26, 'DV': 25, 'Vaccinia': 25, 'Influenza': 19, 'SIV': 10, 'SARS-CoV': 10}


def make_seq_list(path):
    seq_list = []

    with open(path) as f:
        for line in f:
            seq = line[:-1]
            seq_list.append(seq)
    return seq_list


def calc_best_alignment(synseq, realseq_list):
    max_score = -1*math.inf
    sim_seq = ''
    best_alignment = None

    for realseq in realseq_list:
        if len(realseq) > 50:  # seq with length over 50 excluded
            continue
        alignments = pairwise2.align.globalms(
            synseq, realseq, 2, -1, -1, -.5)
        if max(max_score, alignments[0][2]) == alignments[0][2]:
            max_score = alignments[0][2]  # alignment score
            sim_seq = realseq
            best_alignment = alignments[0]
    return max_score, sim_seq, best_alignment


def out_result(seq_list, train_seq_v_dic, val_seq_v_dic, align_path):
    '''
    make pairwise alignment with real data
    '''
    realseq_list = list(train_seq_v_dic.keys())
    realseq_list += list(val_seq_v_dic.keys())

    seq_set = set(seq_list)

    with open(align_path, 'w') as f:
        result_list = []
        for synseq in seq_set:
            max_score, sim_seq, best_alignment = calc_best_alignment(
                synseq, realseq_list)
            # print(format_alignment(*best_alignment))
            f.write(''.join([format_alignment(*best_alignment), '\n']))

            tv = 'train' if sim_seq in train_seq_v_dic else 'val'
            sim_virus = train_seq_v_dic[sim_seq] if tv == 'train' else val_seq_v_dic[sim_seq]

            result_list.append(
                [synseq, sim_seq, max_score, sim_virus, tv])

    result_df = pd.DataFrame(result_list, columns=column_idx)
    result_df = result_df.set_index('Synthetic Sequence')

    return result_df


def analyse_result(result_df, analyse_path):
    score_ave = result_df[column_idx[2]].mean()  # Score Target
    score_stdev = result_df[column_idx[2]].std()  # Score Target
    tt_dic = result_df[column_idx[4]].value_counts().to_dict()  # Train or Val
    virus_dic = result_df[column_idx[3]
                          ].value_counts().to_dict()  # Target Virus
    with open(analyse_path, 'w') as f:
        # f.write(
        #     'Score Average\tScore Standard Deviation\tTrain/Test Dictionary\tVirus Dictionary\n')
        f.write('\t'.join(['Score Average', str(score_ave), '\n']))
        f.write(
            '\t'.join(['Score Standard Deviation', str(score_stdev), '\n\n']))

        list_dic = [tt_dic, virus_dic, tvposv_dic]

        for dic in list_dic:
            for k, v in dic.items():
                f.write('\t'.join([k, str(v), '\n']))
            f.write('\n')


def pairwise_main(input_dir, train_seq_v_dic, val_seq_v_dic, out_path, analyse_path):
    seq_list = make_seq_list(input_dir)
    result_df = out_result(seq_list, train_seq_v_dic, val_seq_v_dic, out_path)
    analyse_result(result_df, analyse_path)
    result_df = result_df.astype(str)
    return result_df
