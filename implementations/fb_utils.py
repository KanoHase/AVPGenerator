import os
import numpy as np
import time

input_path= './data_fbgan/input/'
output_path='./data_fbgan/output/'

def select_pos_seqs(sampled_seqs, epoch, preds_cutoff):
    make_input_file(sampled_seqs, epoch)
    
    preds = make_pred(epoch)
    good_indices = (preds > preds_cutoff).nonzero()[0]
    pos_seqs = [list(sampled_seqs[i]) for i in good_indices]
    return pos_seqs

def make_input_file(sampled_seqs, epoch):
    fasta_seqs=[('>'+str(i)+'\n'+seq+'\n') for (i,seq) in enumerate(sampled_seqs)]
    joint_fasta_seqs=''.join(fasta_seqs)
    
    with open (input_path+'input_{0}.txt'.format(epoch), mode='w') as f:
        f.write(joint_fasta_seqs)

def make_pred(epoch):
    while True:
        if os.path.exists(output_path+'output_'+str(epoch)+'.txt'):
            break
        else:
            time.sleep(1)

    with open (output_path+'output_{0}.txt'.format(epoch)) as f:
        pred_tmp=f.read()

    pred_list_str=pred_tmp.split('\n')
    del pred_list_str[-1]

    pred_list = [[float(val)] for val in pred_list_str]
    pred = np.array(pred_list, dtype='float')
    return pred