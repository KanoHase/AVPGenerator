import subprocess
import os
import Levenshtein
import statistics
import numpy as np
from scipy import stats
import pandas as pd

HOME = os.environ['HOME']
alphafold_dir = HOME + '/Desktop/hdd/AVPGenerator/alphafold_out/'


def dssp_out(path):
    target = '#'
    # helix (G, H and I), strand (E and B) and loop (S, T, and C)
    letter_list = ['G', 'H', 'I', 'E', 'B', 'S', 'T', 'C']

    struc = ''
    seq = ''

    if not os.path.exists(path+"/ranked_0.pdb"):
        print('STRUCTURE UNPREDICTED:', path)

    else:
        cmd = "mkdssp -i {0}/ranked_0.pdb".format(path)
        res = subprocess.check_output(
            cmd, shell=True)
        res = str(res)

        idx = res.find(target)
        table = res[idx+1:]  # getting str after '#'
        list_tmp = table.split('\\n')  # first splitting with '\\n'
        del list_tmp[-1]  # last ',' is unnecessary

        table_array = [row.split() for row in list_tmp]
        del table_array[0]  # first one (index) is unnecessary

        for a in table_array:
            if a[4] in letter_list:
                struc += a[4]
            else:
                struc += 'C'
            seq += a[3]

    return struc, seq


def dev_numdic(path):
    with open(path) as f:
        numdic = {}
        f_content = f.read()
        for line in f_content.split("\n"):
            if line.startswith(">"):
                val_tmp = line.rstrip('\n').replace(">", "")
            elif not line.rstrip('\n') == '':
                numdic[line.rstrip('\n')] = val_tmp
    return numdic


def lis2df(tarseq_lis, tarstruc_lis, comseq_lis, comstruc_lis, dist_lis, tarnum_lis, comnum_lis, tv_lis, vir_dis, target_name, compare_name):
    pair_dist_dic = {}
    pair_dist_dic[target_name+' Sequence'] = tarseq_lis
    pair_dist_dic[target_name+' Structure'] = tarstruc_lis
    pair_dist_dic[compare_name+' Sequence'] = comseq_lis
    pair_dist_dic[compare_name+' Structure'] = comstruc_lis
    pair_dist_dic['Levenstein Distance'] = dist_lis
    pair_dist_dic['Train or Val'] = tv_lis
    pair_dist_dic['Possible Virus'] = vir_dis

    if target_name == 'Synthetic':
        pair_dist_dic[target_name+' num'] = tarnum_lis
        pair_dist_dic[compare_name+' num'] = comnum_lis

    pair_dist_df = pd.DataFrame(pair_dist_dic)
    return pair_dist_df


def calc_levdist(tarstruc, compdic):
    min_dist = 100000
    for posk, posv in compdic.items():
        dist = Levenshtein.distance(tarstruc, posv)
        if dist <= min_dist:
            min_dist = dist
            comseq, comstruc = posk, posv
    return comseq, comstruc, min_dist


def make_df(struc_dir_dic, target_dir, compare_dir, target_name, compare_name, train_seq_v_dic, val_seq_v_dic):

    tarseq_lis, tarstruc_lis, comseq_lis, comstruc_lis, dist_lis, tarnum_lis, comnum_lis, tv_lis, vir_dis = [
    ], [], [], [], [], [], [], [], []

    if target_name == 'Synthetic':
        tarnumdic = dev_numdic(alphafold_dir+'syn_all.fasta')
        comnumdic = dev_numdic(alphafold_dir+'pos_all.fasta')

    for tarseq, tarstruc in struc_dir_dic[target_dir].items():
        if tarseq == '':
            continue

        comseq, comstruc, min_dist = calc_levdist(
            tarstruc, struc_dir_dic[compare_dir])

        tv = 'train' if comseq in train_seq_v_dic else 'val'
        sim_virus = train_seq_v_dic[comseq] if tv == 'train' else val_seq_v_dic[comseq]

        if target_name == 'Synthetic':
            tarnum, comnum = tarnumdic[tarseq], comnumdic[comseq]

        tarseq_lis.append(tarseq)
        tarstruc_lis.append(tarstruc)
        comseq_lis.append(comseq)
        comstruc_lis.append(comstruc)
        dist_lis.append(min_dist)
        tv_lis.append(tv)
        vir_dis.append(sim_virus)

        if target_name == 'Synthetic':
            tarnum_lis.append(tarnum)
            comnum_lis.append(comnum)

    pair_dist_df = lis2df(tarseq_lis, tarstruc_lis,
                          comseq_lis, comstruc_lis, dist_lis, tarnum_lis, comnum_lis, tv_lis, vir_dis, target_name, compare_name)
    return pair_dist_df


def write_analyse_dist(syn_mean, neg_mean, syn_stdev, neg_stdev, yu, tt_dic, virus_dic, path):
    tvposv_dic = {'HIV': 279, 'HSV': 61, 'RSV': 57, 'FIV': 51, 'MV': 49, 'HCV': 45, 'HPF3': 37, 'WNV': 30,
                  'paramyxoviruses': 26, 'DV': 25, 'Vaccinia': 25, 'Influenza': 19, 'SIV': 10, 'SARS-CoV': 10}

    with open(path, 'w') as f:
        f.write('\tSyn\tNeg\n')
        f.write('\t'.join(['Average', str(syn_mean), str(neg_mean), '\n']))
        f.write('\t'.join(['Standard Deviation', str(
            syn_stdev), str(neg_stdev), '\n']))
        f.write('\t'.join(['mannwhitneyu', str(yu), '\n\n']))

        list_dic = [tt_dic, virus_dic, tvposv_dic]

        for dic in list_dic:
            for k, v in dic.items():
                f.write('\t'.join([k, str(v), '\n']))
            f.write('\n')


def analyse_from_df(syn_pair_dist_df, neg_pair_dist_df, path):
    syn_distlis, neg_distlis = syn_pair_dist_df['Levenstein Distance'].to_list(
    ), neg_pair_dist_df['Levenstein Distance'].to_list()

    syn_mean, syn_stdev = statistics.mean(
        syn_distlis), statistics.stdev(syn_distlis)
    neg_mean, neg_stdev = statistics.mean(
        neg_distlis), statistics.stdev(neg_distlis)

    yu = stats.mannwhitneyu(np.array(list(syn_distlis)), np.array(
        list(neg_distlis)), alternative='two-sided')

    tt_dic = syn_pair_dist_df['Train or Val'].value_counts(
    ).to_dict()  # Train or Val
    virus_dic = syn_pair_dist_df['Possible Virus'].value_counts(
    ).to_dict()  # Target Virus
    write_analyse_dist(syn_mean, neg_mean, syn_stdev,
                       neg_stdev, yu, tt_dic, virus_dic, path)


def dssp_main(path, train_seq_v_dic, val_seq_v_dic):
    dirs = ['posdata_out', 'syndata_out', 'negdata_out']
    struc_dir_dic = {}

    for d in dirs:
        struc_seq_dic = {}
        file_list = os.listdir(alphafold_dir+d+'/')

        for file in file_list:
            struc, seq = dssp_out("{0}{1}/{2}/".format(alphafold_dir, d, file))
            struc_seq_dic[seq] = struc
            # struc_seq_dic: {seq(str): corresponding structure(str)}

        struc_dir_dic[d] = struc_seq_dic
        # struc_seq_dic: {dir_name: {seq(str): corresponding structure(str)}}

    syn_pair_dist_df = make_df(
        struc_dir_dic, 'syndata_out', 'posdata_out', 'Synthetic', 'Positive', train_seq_v_dic, val_seq_v_dic)
    neg_pair_dist_df = make_df(
        struc_dir_dic, 'negdata_out', 'posdata_out', 'Negative', 'Positive', train_seq_v_dic, val_seq_v_dic)
    analyse_from_df(syn_pair_dist_df, neg_pair_dist_df, path)
    return syn_pair_dist_df


"""
HOME = os.environ['HOME']
path = HOME + '/Desktop/hdd/AVPGenerator/eval_/ep75_ba64_lr0.0001_pc0.8_optAdam_itr10/similar_struct_analyse.txt'
syn_pair_dist_df = dssp_main(path)
"""
