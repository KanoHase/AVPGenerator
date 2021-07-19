import os
import pandas as pd
import numpy as np
from propy import PyPro
# # check: https://propy3.readthedocs.io/en/latest/PyPro.html
from modlamp.descriptors import GlobalDescriptor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--allepoch", action='store_true',
                    help="choose whether or not you want to evaluate all epoch's transition. Place --allepoch if you need evaluation for all epoch's transition.")
opt = parser.parse_args()

data_dir = "./data/"
samples_dir = "./samples/"
eval_dir = "./eval/"
real_file = "val_positive"
gen_file = "100.txt"
options = ["ut", "fe", "fp", "mp", "rev"]
task_list = ["Length", "Isoelectric Point", "Hydrophobicity"]
std_task_list = ["Standard Deviation of Length",
                 "Standard Deviation of Isoelectric Point", "Standard Deviation of Hydrophobicity"]
if not os.path.exists(eval_dir):
    os.mkdir(eval_dir)


def real_data_process(real_path):
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
        if prop_array == []:
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


def calc_sim_rate(norm_prop_df, threshold):
    sq_norm_prop_df = norm_prop_df**2
    dist_df = np.sqrt(sq_norm_prop_df.sum(axis=1))
    bool_dist_df = (dist_df < threshold)
    sim_rate = bool_dist_df.sum()/len(bool_dist_df.index)*100
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
        row = ["-" for _ in range(len(options))] + [str(round(val, 3))
                                                    for val in real_data] + ["-"]
        f.write("\t".join(row))

        # sort by the last index of the vals of other data
        dic = sorted(dic.items(), key=lambda x: -x[1][len(x[1])-1])

        # write other data
        for k, lis in dic:
            for op in options:
                k = k.replace(op, "")
            k = "\n" + k
            row = k.split("_") + [str(round(val, 3)) for val in lis]

            f.write("\t".join(row))
    return


def main():
    # calculate real data's average and std
    real_ave_std_df, prop_real, std_real = real_data_process(
        data_dir+real_file+".txt")

    run_dirs = os.listdir(samples_dir)

    prop_header = options + task_list + ["Similarity Rate(%)"]
    prop_dic = {}

    std_header = options + std_task_list + \
        ["Average Standard Deviation (Normalized)"]
    std_dic = {}

    for run_dir in run_dirs:
        if opt.allepoch:
            allepoch_transition(samples_dir, eval_dir, task_list, run_dir)

        seq_list = []

        with open(samples_dir+run_dir+"/"+gen_file) as f:
            for line in f:
                seq = line[:-1]
                seq_list.append(seq)

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
        prop_dic[run_dir] = prop_summary_list

        # calculate std of norm prop
        norm_ave_std_df = avestd_from_df(norm_prop_df)  # kari
        # average of std #kari
        ave_of_std = np.mean(
            norm_ave_std_df.loc["Standard Deviation"].values.tolist())

        # rundir_ave_std_df[Length, Isoelectric Point, Hydrophobicity], ave_of_std
        std_summary_list = rundir_ave_std_df.loc["Standard Deviation"].values.tolist(
        )
        std_summary_list.append(ave_of_std)
        std_dic[run_dir] = std_summary_list

        # calculate and write each run_dir/100.txt's aacomp
        if not os.path.exists(eval_dir+run_dir+"/"):
            os.mkdir(eval_dir+run_dir+"/")
        calc_write_aacomp(seq_list, eval_dir+run_dir+"/aacomp.txt")

    dic_to_file(prop_header, prop_real, options,
                prop_dic, eval_dir+"/properties.txt")
    dic_to_file(std_header, std_real, options,
                std_dic, eval_dir+"/stdev.txt")


if __name__ == '__main__':
    main()
