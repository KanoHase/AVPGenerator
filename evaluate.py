from propy import PyPro
# check: https://propy3.readthedocs.io/en/latest/PyPro.html
import argparse
import os
from implementations.afterprocess import makehist_from_intlist, makehist_from_diclist

parser = argparse.ArgumentParser()
parser.add_argument("--nolength", action='store_false',
                    help="choose whether or not you want to evaluate length. Place --nolengths if you don't need length for evaluation.")
parser.add_argument("--noaacomp", action='store_false',
                    help="choose whether or not you want to evaluate aacomp. Place --noaacomp if you don't need aacomp for evaluation.")
parser.add_argument("--noctd", action='store_false',
                    help="choose whether or not you want to evaluate ctd. Place --noctd if you don't need ctd for evaluation.")
opt = parser.parse_args()
do_length = opt.nolength
do_aacomp = opt.noaacomp
do_ctd = opt.noctd

data_dir = "./data/"
samples_dir = "./samples/"
eval_dir = "./eval/"
real_file = "positive_540"
generated_file = "samples"
eval_summary = "eval_summary"
if not os.path.exists(eval_dir):
    os.makedir(eval_dir)

file_list = [[data_dir, real_file], [samples_dir, generated_file]]
row_list = []

with open(eval_dir + eval_summary + ".txt", "w") as g:
    for dir_name, file_name in file_list:
        with open(dir_name + file_name + ".txt") as f:
            len_list = []
            aacomp_diclist = []
            ctd_diclist = []
            header_list = []
            # row_list.append("\n")
            row_list.append("\n"+file_name)

            for line in f:
                seq = line[:-1]
                DesObject = PyPro.GetProDes(seq)
                if do_length:
                    len_list.append(len(seq))
                if do_aacomp:
                    aacomp_diclist.append(DesObject.GetAAComp())
                if do_ctd:
                    # calculate 147 CTD descriptors
                    ctd_diclist.append(DesObject.GetCTD())

        if do_length:
            len_ave = makehist_from_intlist(
                len_list, "len", eval_dir, file_name)
            header_list.append("Length")
            row_list.append(str(len_ave))
            # print(len_ave, eval_dir, file_name, dir_name)
        if do_aacomp:
            aacomp_ave_dic = makehist_from_diclist(
                aacomp_diclist, "aacomp", eval_dir, file_name)
            for k in aacomp_ave_dic.keys():
                header_list.append(k)
                row_list.append(str(aacomp_ave_dic[k]))
        if do_ctd:
            ctd_ave_dic = makehist_from_diclist(
                ctd_diclist, "ctd", eval_dir, file_name)
            for k in ctd_ave_dic.keys():
                header_list.append(k)
                row_list.append(str(ctd_ave_dic[k]))

    header_list.insert(0, " ")
    g.write("\t".join(header_list))
    g.write("\t".join(row_list))
