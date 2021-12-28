import os
import glob
import re
import pandas as pd

docking_dir = './docking/'
dtype_list = ['pos', 'neg', 'syn']
col_index = ['trajectory (replica) index',
             'trajectory (replica) frame index',
             'temperature',
             'energy for protein receptor',
             'energy for peptide',
             'protein-peptide interaction energy',
             'protein-peptide total energy']
transition_file = 'transition.txt'
ene_summary_file = 'energy_summary.txt'
tra_summary_dic = {}
ene_summary_dic = {}

for dtype in dtype_list:
    path = docking_dir+dtype+'_docking/'
    dirlis = os.listdir(path)

    for dir in dirlis:
        if '.' not in dir:
            name = path+dir+'/*'
            # any dir name is fine bc there's only one dir (starts with CABS)
            target_path = glob.glob(name)
            df = pd.read_csv(target_path[0]+'/energy.txt', sep="\s+")
            # setting column names
            df = df.set_axis(col_index, axis='columns')
            # setting the very left column as index
            df = df.set_index('trajectory (replica) index')

            min_ene = 100
            tra = 1

            for i in range(1, 11):
                idx = re.escape(str(i))+r'$'  # matching index name with i
                sdf = df.filter(regex=idx, axis=0)  # df with the same index
                mintmp = sdf['protein-peptide interaction energy'].min()
                if min(min_ene, mintmp) == mintmp:
                    min_ene = mintmp
                    min_tra_df = sdf
                    tra = i

            tra_summary_dic[dtype+'_' +
                            dir] = min_tra_df['protein-peptide interaction energy']
            ene_summary_dic[dtype+'_'+dir] = [tra, min_ene]


with open(docking_dir+transition_file, 'w') as f:
    for key, val in tra_summary_dic.items():
        val_list_float = val.values.tolist()  # to list
        val_list = [str(num) for num in val_list_float]  # to str list
        row = [key] + val_list + ['\n']
        f.write('\t'.join(row))

with open(docking_dir+ene_summary_file, 'w') as f:
    f.write('Pair name\tTrajectory\tEnergy\n')
    for key, val in ene_summary_dic.items():
        f.write('\t'.join([key, str(val[0]), str(val[1]), '\n']))
