#    Copyright (C) 2018 Anvita Gupta
#
#    This program is free software: you can redistribute it and/or  modify
#    it under the terms of the GNU Affero General Public License, version 3,
#    as published by the Free Software Foundation.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import torch
import torch.nn as nn
from utils.bio_utils import *

import pathlib
import os
import numpy as np
import time


class AVPClassifier():
    def __init__(self, dataset='./data/AVP_dataset.fa', run_name='class_pytorch_drop_03'):
        print('Initialized')

    def evaluate_model(self, validation=True):
        return (1,1)

    def predict_model(self, sampled_seqs, epoch, c):

        amino_seqs_tmp, valid_gene_seqs=geneToProtein(sampled_seqs)#sampled:960, aminoseqs:less than 960

        if not valid_gene_seqs:
            return np.array([[0]], dtype='float'), ['']

        amino_seqs=[('>'+str(i)+'\n'+seq+'\n') for (i,seq) in enumerate(amino_seqs_tmp)]
        all_seqs=''.join(amino_seqs)
        
        #current_path=pathlib.Path.cwd()
        input_path= '/home/kano_hasegawa/Dropbox/FBGAN/input/'
        output_path='/home/kano_hasegawa/Dropbox/FBGAN/output/'

        #if os.path.exists(output_path+'output_'+str(epoch)+'.txt'):
        with open (input_path+'input_{0}_{1}.txt'.format(epoch,c), mode='w') as f:
            f.write(all_seqs)
        with open (input_path+'epoch.txt', mode='w') as g:
            g.write(str(epoch)+'_'+str(c))

        while True:
            if os.path.exists(output_path+'output_'+str(epoch)+'_'+str(c)+'.txt'):
                break
            else:
                time.sleep(1)

        with open (output_path+'output_{0}_{1}.txt'.format(epoch,c)) as f:
            pred_tmp=f.read()

        pred_list_str=pred_tmp.split('\n')
        del pred_list_str[-1]

        pred_list=[[float(val)] for val in pred_list_str]
        pred=np.array(pred_list, dtype='float')
        # print(len(pred), len(valid_gene_seqs))

        return pred, valid_gene_seqs

        #else:
            #print("Failed loading data...")

            #return np.array([[0]], dtype='float'),['']


if __name__ == '__main__':
    main()
