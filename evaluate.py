from propy import PyPro
# check: https://propy3.readthedocs.io/en/latest/PyPro.html
from modlamp.descriptors import GlobalDescriptor
import argparse
import os
from implementations.afterprocess import makehist_from_intlist, makehist_from_diclist, makeintlistdic_from_allep, makeplot_from_intlistdic, makesummary_from_file, maketsne_from_file
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--nolength", action='store_false',
                    help="choose whether or not you want to evaluate length. Place --nolengths if you don't need length for evaluation.")
parser.add_argument("--noaacomp", action='store_false',
                    help="choose whether or not you want to evaluate aacomp. Place --noaacomp if you don't need aacomp for evaluation.")
parser.add_argument("--nopi", action='store_false',
                    help="choose whether or not you want to evaluate pi. Place --nopi if you don't need aacomp for evaluation.")
parser.add_argument("--nohyd", action='store_false',
                    help="choose whether or not you want to evaluate aacomp. Place --nohyd if you don't need aacomp for evaluation.")
parser.add_argument("--ctd", action='store_true',
                    help="choose whether or not you want to evaluate ctd. Place --ctd if you need ctd for evaluation.")
parser.add_argument("--nolenallep", action='store_false',
                    help="choose whether or not you want to evaluate length for all epoch. Place --nolenallep if you don't need lenallep for evaluation.")
parser.add_argument("--novar", action='store_false',
                    help="choose whether or not you want to evaluate variance for all epoch. Place --novar if you don't need var for evaluation.")
parser.add_argument("--tsne", action='store_true',
                    help="choose whether or not you want to create tsne. Place --tsne if you need tsne for evaluation.")
parser.add_argument("--last_ep", type=int, default=10,
                    help="")
opt = parser.parse_args()
do_length = opt.nolength
do_aacomp = opt.noaacomp
do_pi = opt.nopi
do_hyd = opt.nohyd
do_ctd = opt.ctd
do_plot_allep = opt.nolenallep
do_var = opt.novar
do_tsne = opt.tsne


data_dir = "./data/"
samples_dir = "./samples/"
eval_dir = "./eval/"
real_file = "val_positive"
generated_file = "100"  # kari
eval_summary = "property_summary"
if not os.path.exists(eval_dir):
    os.mkdir(eval_dir)
# if os.path.exists(eval_dir+eval_summary+".txt"):
#     with open (eval_dir+eval_summary+".txt") as f:
#         for line in f:


file_list = [[data_dir, "", real_file]]
run_dirs = os.listdir(samples_dir)
existing_run_dirs = os.listdir(eval_dir)

run_dirs_df = pd.DataFrame(run_dirs)
existing_run_dirs_df = pd.DataFrame(existing_run_dirs)
new_run_dirs_df = run_dirs_df[~run_dirs_df[0].isin(existing_run_dirs_df[0])]
new_run_dirs = new_run_dirs_df[0].values.tolist()

for run_dir in run_dirs:
    if not os.path.exists(samples_dir + run_dir+"/" + generated_file + ".txt"):
        file_name = "99"  # kari
    else:
        file_name = generated_file
    file_list.append([samples_dir, run_dir+"/", file_name])

row_list = []

if do_tsne:
    maketsne_from_file(file_list, eval_dir)


with open(eval_dir + eval_summary + ".txt", "w") as g:
    for dir_name, run_dir, file_name in file_list:
        with open(dir_name + run_dir + file_name + ".txt") as f:
            len_list = []
            aacomp_diclist = []
            pi_list = []
            hyd_list = []
            ctd_diclist = []
            header_list = []
            # row_list.append("\n")
            row_list.append("\n" + run_dir + file_name)

            for line in f:
                seq = line[:-1]
                DesObject = PyPro.GetProDes(seq)
                if do_length:
                    len_list.append(len(seq))
                if do_pi:
                    glob_seq = GlobalDescriptor(seq)
                    glob_seq.isoelectric_point()
                    pi_list.append(glob_seq.descriptor[0][0])
                if do_hyd:
                    glob_seq = GlobalDescriptor(seq)
                    glob_seq.hydrophobic_ratio()
                    hyd_list.append(glob_seq.descriptor[0][0])
                if do_aacomp:
                    aacomp_diclist.append(DesObject.GetAAComp())
                if do_ctd:
                    # calculate 147 CTD descriptors
                    # Default: False
                    ctd_diclist.append(DesObject.GetCTD())

        if file_name == real_file:
            run_dir = real_file + "/"
        if not os.path.exists(dir_name + run_dir):
            os.mkdir(dir_name + run_dir)
        if do_length:
            len_ave = makehist_from_intlist(
                len_list, "len", eval_dir, run_dir, file_name)
            header_list.append("Length")
            row_list.append(str(len_ave))
        if do_pi:
            pi_ave = makehist_from_intlist(
                pi_list, "pi", eval_dir, run_dir, file_name)
            header_list.append("PI")
            row_list.append(str(pi_ave))
        if do_hyd:
            hyd_ave = makehist_from_intlist(
                hyd_list, "hyd", eval_dir, run_dir, file_name)
            header_list.append("Hydrophobicity")
            row_list.append(str(hyd_ave))
        if do_aacomp:
            aacomp_ave_dic = makehist_from_diclist(
                aacomp_diclist, "aacomp", eval_dir, run_dir, file_name)
            for k in aacomp_ave_dic.keys():
                header_list.append(k)
                row_list.append(str(aacomp_ave_dic[k]))
        if do_ctd:  # Default: false
            ctd_ave_dic = makehist_from_diclist(
                ctd_diclist, "ctd", eval_dir, run_dir, file_name)
            for k in ctd_ave_dic.keys():
                header_list.append(k)
                row_list.append(str(ctd_ave_dic[k]))
        if do_plot_allep and (file_name != real_file) and (run_dir in new_run_dirs):
            int_list_dic = makeintlistdic_from_allep(
                dir_name, run_dir)
            makeplot_from_intlistdic(int_list_dic, eval_dir, run_dir)

        print("DONE: ", run_dir)

    header_list.insert(0, " ")
    g.write("\t".join(header_list))
    g.write("\t".join(row_list))

    if do_var:
        var_file_list = ["len_var_allepoch",
                         "pi_var_allepoch", "hyd_var_allepoch"]
        makesummary_from_file(eval_dir, run_dirs, var_file_list, opt.last_ep)
