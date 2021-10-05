import os
import pandas as pd
import numpy as np
from propy import PyPro
# check: https://propy3.readthedocs.io/en/latest/PyPro.html
from modlamp.descriptors import GlobalDescriptor
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--allepoch", action='store_true',
                    help="choose whether or not you want to evaluate all epoch's transition. Place --allepoch if you need evaluation for all epoch's transition.")
parser.add_argument("--nocompareprop", action='store_true',
                    help="choose whether or not you want to compare properties with real data. Place --nocompareprop if you don't need it.")
opt = parser.parse_args()

chosen_dir = "ep75_ba64_lr0.0001_pc0.8_optAdam_itr10/"
# kari # use this to compare properties with real data
opt_name = "posscreen_negexpnoexp"
data_dir = "./real_data_"+opt_name+"/"
raw_dir = "./raw_data/"
samples_dir = "./samples/"
eval_dir = "./eval/"
real_pos_file = "val_positive"
real_pos_train_file = "positive"
real_neg_file = "val_negative"
simprop_file = "prop_comparison"
options = ["ep", "ba", "lr", "pc", "opt", "itr"]
query = 'ep(.*)_ba'
task_list = ["Length", "Isoelectric Point", "Hydrophobicity"]
std_task_list = ["Standard Deviation of Length",
                 "Standard Deviation of Isoelectric Point", "Standard Deviation of Hydrophobicity"]
used_v_list = ['HIV', 'HSV', 'RSV', 'FIV', 'MV', 'HCV', 'HPF3', 'WNV',
               'paramyxoviruses', 'DV', 'Vaccinia', 'Influenza', 'SIV', 'SARS-CoV']
if not os.path.exists(eval_dir):
    os.mkdir(eval_dir)


def real_data_process(real_path, real_file):
    real_seq_list = []

    with open(real_path) as f:
        for line in f:
            seq = line[:-1]
            real_seq_list.append(seq)
    prop_df = calc_prop(real_seq_list, task_list)
    real_ave_std_df = avestd_from_df(prop_df)

    prop_real = real_ave_std_df.loc["Average"].values.tolist()
    std_real = real_ave_std_df.loc["Standard Deviation"].values.tolist()

    if not os.path.exists(eval_dir+real_file+"/"):
        os.mkdir(eval_dir+real_file+"/")
    calc_write_aacomp(real_seq_list, eval_dir+real_file+"/aacomp.txt")

    return real_ave_std_df, prop_real, std_real


def make_seq_list(path, raw=False):
    seq_list = []

    if raw:
        df = pd.read_csv(path, sep="\t")
        seq_v_dic = dict(zip(df['Sequence'], df['Virus']))
        seq_list = list(seq_v_dic.keys())
    else:
        with open(path) as f:
            for line in f:
                seq = line[:-1]
                seq_list.append(seq)
        seq_v_dic = None
    return seq_list, seq_v_dic


def calc_prop(seq_list, task_list):
    prop_array = []
    glob_seq = GlobalDescriptor(seq_list)

    for task in task_list:
        if task == "Length":
            glob_seq.length()
        if task == "Isoelectric Point":
            glob_seq.isoelectric_point()
        if task == "Hydrophobicity":
            glob_seq.hydrophobic_ratio()
        if len(prop_array) == 0:
            prop_array = glob_seq.descriptor
        else:
            prop_array = np.concatenate([prop_array, glob_seq.descriptor], 1)
    prop_df = pd.DataFrame(prop_array, columns=task_list)
    return prop_df


def norm_prop(prop_df, ave_std_df):
    norm_prop_df = (
        prop_df - ave_std_df.loc["Average"]) / ave_std_df.loc["Standard Deviation"]  # calculate normalised prop
    return norm_prop_df


def avestd_from_df(prop_df):
    ave_df = prop_df.mean()
    std_df = prop_df.std()
    ave_std_df_tmp = pd.concat([ave_df, std_df], axis=1)
    ave_std_df = ave_std_df_tmp.set_axis(
        ["Average", "Standard Deviation"], axis=1)
    return ave_std_df.T


def var_from_df(prop_df):
    var_df = prop_df.var()
    return var_df


def calc_sim_rate(norm_prop_df, threshold):
    sq_norm_prop_df = norm_prop_df**2
    dist_df = np.sqrt(sq_norm_prop_df.sum(axis=1))
    bool_dist_df = (dist_df < threshold)
    sim_rate = bool_dist_df.sum()/len(bool_dist_df.index)
    return sim_rate


def calc_write_aacomp(seq_list, path):
    for i, seq in enumerate(seq_list):
        DesObject = PyPro.GetProDes(seq)
        if i == 0:
            aacomp_dic = DesObject.GetAAComp()
        else:
            for k in DesObject.GetAAComp().keys():
                aacomp_dic[k] += DesObject.GetAAComp()[k]

    with open(path, "w") as f:
        for k in aacomp_dic.keys():
            val = round(aacomp_dic[k]/len(seq_list), 3)
            row = [k, str(val), "\n"]
            f.write("\t".join(row))


def allepoch_transition(samples_dir, eval_dir, task_list, run_dir):
    i = 1
    ave_list = []
    std_list = []

    while True:
        if os.path.exists(samples_dir+run_dir+"/"+str(i)+'.txt'):
            with open(samples_dir+run_dir+"/"+str(i)+".txt") as f:
                seq_list = []
                for line in f:
                    seq = line[:-1]
                    seq_list.append(seq)

                prop_df = calc_prop(seq_list, task_list)
                ave_std_df = avestd_from_df(prop_df)

                ave_list_tmp = ave_std_df.loc['Average'].values.tolist()
                std_list_tmp = ave_std_df.loc['Standard Deviation'].values.tolist(
                )

                ave_list.append(ave_list_tmp)
                std_list.append(std_list_tmp)

            i += 1

        else:
            break

    write_from_list(ave_list, eval_dir+run_dir +
                    "/average_allepoch.txt", task_list)
    write_from_list(std_list, eval_dir+run_dir +
                    "/std_allepoch.txt", std_task_list)
    print("DONE: ", run_dir)


def write_from_list(lists, path, std_task_list):
    """
    INPUT
    lists:list(list)
    """

    with open(path, "w") as f:
        f.write("\t".join(std_task_list))
        f.write("\n")

        for lis in lists:
            lis = [str(round(l, 3)) for l in lis]
            f.write("\t".join(lis))
            f.write("\n")


def dic_to_file(header, real_data, options, dic, path):
    """
    INPUT
    header:list(str)
    real_data:list(float)
    dic: dic(index:list) e.g.{"utP-PS...": [], ...}
    """

    with open(path, "w") as f:
        # write header
        f.write("\t".join(header))
        f.write("\n")

        # write real data
        row = [real_pos_file.replace("_", "-")] + ["-" for _ in range(len(options)-1)] + [str(round(val, 3))
                                                                                          for val in real_data] + ["-"]
        f.write("\t".join(row))

        # sort by the last index of the vals of other data
        dic = sorted(dic.items(), key=lambda x: -x[1][len(x[1])-1])

        # write other data
        for k, lis in dic:
            if k == real_neg_file:
                f.write('\n')
                row = [real_neg_file.replace("_", "-")] + ["-" for _ in range(len(options)-1)] + \
                    [str(round(val, 3)) for val in lis]
            else:
                for op in options:
                    k = k.replace(op, "")
                k = "\n" + k
                row = k.split("_") + [str(round(val, 3)) for val in lis]

            f.write("\t".join(row))
    return


def make_prop_dic(prop_dic, seq_list, real_ave_std_df, dir_name):
    # calculate each run_dir/100.txt's prop
    prop_df = calc_prop(seq_list, task_list)

    # normalize each run_dir/100.txt's prop using real average and std
    norm_prop_df = norm_prop(prop_df, real_ave_std_df)

    # gives distance between gen and real prop
    sim_rate = calc_sim_rate(norm_prop_df, 1.5)

    # calculate each run_dir/100.txt's prop's average and std
    rundir_ave_std_df = avestd_from_df(prop_df)

    # rundir_ave_std_df[Length, Isoelectric Point, Hydrophobicity], sim_rate
    prop_summary_list = rundir_ave_std_df.loc["Average"].values.tolist(
    ) + [sim_rate]
    prop_dic[dir_name] = prop_summary_list
    return prop_dic, norm_prop_df, rundir_ave_std_df


def make_std_dic(std_dic, norm_prop_df, rundir_ave_std_df, dir_name):
    # calculate variance of norm prop
    var_df = var_from_df(norm_prop_df)

    # average of variance
    ave_of_var = np.mean(var_df.values.tolist())

    # rundir_ave_std_df[Length, Isoelectric Point, Hydrophobicity], ave_of_var
    std_summary_list = rundir_ave_std_df.loc["Standard Deviation"].values.tolist(
    )
    std_summary_list.append(ave_of_var)
    std_dic[dir_name] = std_summary_list
    return std_dic


def make_seqprop_df(path, t_or_v, raw=False):
    '''
    for compareprop only
    '''
    seq_list, seq_v_dic = make_seq_list(path, raw=raw)
    prop_df = calc_prop(seq_list, task_list)
    seqprop_df = pd.concat(
        [pd.DataFrame(seq_list).set_axis(['Sequence'], axis=1), prop_df], axis=1)
    seqprop_df = pd.concat(
        [seqprop_df, pd.DataFrame([t_or_v]*len(seq_list)).set_axis(['Train or Val'], axis=1)], axis=1)
    return seqprop_df, seq_v_dic


def normdf_for_comparison(all_seqprop_df, ave_std_df):
    '''
    for compareprop only
    '''
    all_seqprop_df = all_seqprop_df.set_index('Sequence')
    norm_all_seqprop_df = norm_prop(
        all_seqprop_df.iloc[:, :-1], ave_std_df)
    norm_all_seqprop_df = pd.concat(
        [norm_all_seqprop_df, all_seqprop_df['Train or Val']], axis=1)
    return norm_all_seqprop_df


def calc_all_realdata_normpropdf(real_train_path, real_val_path):
    '''
    for compareprop only
    '''
    train_seqprop_df, train_seq_v_dic = make_seqprop_df(
        real_train_path, 'train', raw=True)
    val_seqprop_df, val_seq_v_dic = make_seqprop_df(
        real_val_path, 'val', raw=True)
    all_real_seqprop_df = pd.concat(
        [train_seqprop_df, val_seqprop_df], axis=0)
    # all_real_seqprop_df = all_real_seqprop_df.set_index('Sequence')
    ave_std_df = avestd_from_df(all_real_seqprop_df.iloc[:, :-1])
    norm_all_real_seqprop_df = normdf_for_comparison(
        all_real_seqprop_df, ave_std_df)
    seq_v_dic = dict(**train_seq_v_dic, **val_seq_v_dic)

    return norm_all_real_seqprop_df, ave_std_df, seq_v_dic


def calc_all_syndata_normpropdf(query, chosen_dir, samples_dir, ave_std_df):
    '''
    for compareprop only
    '''
    name = re.findall(query, chosen_dir)
    gen_file = name[0]+'.txt'

    all_fake_seqprop_df, _ = make_seqprop_df(
        samples_dir+chosen_dir+gen_file, 'syn')

    norm_all_fake_seqprop_df = normdf_for_comparison(
        all_fake_seqprop_df, ave_std_df)
    return norm_all_fake_seqprop_df


def main():
    # calculate real pos and neg data's average and std
    real_ave_std_df, prop_real, std_real = real_data_process(
        data_dir+real_pos_file+".txt", real_pos_file)

    run_dirs = os.listdir(samples_dir)

    prop_header = options + task_list + ["Similarity Rate(%)"]
    prop_dic = {}

    std_header = options + std_task_list + \
        ["Average Variance (Normalized)"]
    std_dic = {}

    for run_dir in run_dirs:
        name = re.findall(query, run_dir)
        gen_file = name[0]+'.txt'

        print(run_dir, gen_file)

        if opt.allepoch:
            allepoch_transition(samples_dir, eval_dir, task_list, run_dir)

        gen_dir = samples_dir+run_dir+"/"+gen_file

        seq_list, _ = make_seq_list(gen_dir)

        prop_dic, norm_prop_df, rundir_ave_std_df = make_prop_dic(
            prop_dic, seq_list, real_ave_std_df, run_dir)

        std_dic = make_std_dic(std_dic, norm_prop_df,
                               rundir_ave_std_df, run_dir)

        # calculate and write each run_dir/100.txt's aacomp
        if not os.path.exists(eval_dir+run_dir+"/"):
            os.mkdir(eval_dir+run_dir+"/")
        calc_write_aacomp(seq_list, eval_dir+run_dir+"/aacomp.txt")

    # real negative data
    seq_list, _ = make_seq_list(data_dir+real_neg_file+".txt")

    prop_dic, norm_prop_df, rundir_ave_std_df = make_prop_dic(
        prop_dic, seq_list, real_ave_std_df, real_neg_file)

    std_dic = make_std_dic(std_dic, norm_prop_df,
                           rundir_ave_std_df, real_neg_file)

    # write results
    dic_to_file(prop_header, prop_real, options,
                prop_dic, eval_dir+"/properties.txt")
    dic_to_file(std_header, std_real, options,
                std_dic, eval_dir+"/stdev.txt")

    # compare each seqs' prop similarity
    if not opt.nocompareprop:
        # real data
        norm_all_real_seqprop_df, ave_std_df, seq_v_dic = calc_all_realdata_normpropdf(
            raw_dir+real_pos_train_file+'.txt', raw_dir+real_pos_file+'.txt')
        print(norm_all_real_seqprop_df)

        # syn data
        norm_all_fake_seqprop_df = calc_all_syndata_normpropdf(
            query, chosen_dir, samples_dir, ave_std_df)
        print(norm_all_fake_seqprop_df)

        with open(eval_dir+chosen_dir+simprop_file+'.txt', 'w') as f:
            f.write(
                'Synthetic Sequence\tMost Similar Real Sequence\tDistance\tTarget Virus\tTrain or Val\n')
            unused_v_list = []

            for index, row in norm_all_fake_seqprop_df.iterrows():
                # print(row.iloc[:-1])
                df_diff_sqr = (
                    row.iloc[:-1]-norm_all_real_seqprop_df.iloc[:, :-1])**2
                dist_df = df_diff_sqr.sum(axis=1)
                mindist_seq, dist = dist_df.idxmin(), dist_df.min()
                f.write(
                    '\t'.join([row.name, mindist_seq, str(dist), seq_v_dic[mindist_seq], norm_all_real_seqprop_df.at[mindist_seq, 'Train or Val'], '\n']))
                print(row.name, mindist_seq, dist,
                      seq_v_dic[mindist_seq], norm_all_real_seqprop_df.at[mindist_seq, 'Train or Val'])

                if seq_v_dic[mindist_seq] not in used_v_list:
                    unused_v_list.append(seq_v_dic[mindist_seq])

        print('untrained viruses: ', set(unused_v_list))


if __name__ == '__main__':
    main()
