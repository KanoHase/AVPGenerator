import os
import numpy as np
import time
from FunctionAnalyser import TransClassifier
from implementations.data_utils import load_data_esm, to_dataloader
from esm_master.initialize_esm import gen_repr

input_path = './data_fbgan/input/'
output_path = './data_fbgan/output/'

if not os.path.exists(input_path):
    os.makedirs(input_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)


def prepare_FA(fbtype, in_dim, out_dim, hidden, batch):
    if fbtype == "Transformer":
        FA = TransClassifier(in_dim, out_dim, hidden, batch)
    else:
        FA = []
    return FA


def trans_select_pos(sampled_seqs, FA, preds_cutoff):
    data_esm = load_data_esm(sampled_seqs)
    seq_repr = gen_repr(data_esm)
    all_preds = FA.analyse_function(seq_repr)
    good_indices = (all_preds > preds_cutoff).nonzero()[0]
    pos_seqs = [list(sampled_seqs[i]) for i in good_indices]
    return pos_seqs


def meta_select_pos(sampled_seqs, epoch, preds_cutoff):
    make_input_file(sampled_seqs, epoch)

    preds = make_pred(epoch)
    good_indices = (preds > preds_cutoff).nonzero()[0]
    pos_seqs = [list(sampled_seqs[i]) for i in good_indices]
    return pos_seqs


def make_input_file(sampled_seqs, epoch):
    fasta_seqs = [('>'+str(i)+'\n'+seq+'\n')
                  for (i, seq) in enumerate(sampled_seqs)]
    joint_fasta_seqs = ''.join(fasta_seqs)

    with open(input_path+'input_{0}.txt'.format(epoch), mode='w') as f:
        f.write(joint_fasta_seqs)


def make_pred(epoch):
    while True:
        if os.path.exists(output_path+'output_'+str(epoch)+'.txt'):
            break
        else:
            time.sleep(1)

    with open(output_path+'output_{0}.txt'.format(epoch)) as f:
        pred_tmp = f.read()

    pred_list_str = pred_tmp.split('\n')
    del pred_list_str[-1]

    pred_list = [[float(val)] for val in pred_list_str]
    pred = np.array(pred_list, dtype='float')
    return pred


def update_data(pos_nparr, seq_nparr, order_label, label_nparr, epoch):
    num_to_add = len(pos_nparr)
    seq_nparr, order_label = remove_old_seq(order_label, seq_nparr, num_to_add)
    #print(seq_nparr.shape, pos_nparr.shape)
    if len(pos_nparr) > 0:
        seq_nparr = np.concatenate([seq_nparr, pos_nparr])
    else:
        seq_nparr = seq_nparr
    order_label = np.concatenate(
        [order_label, np.repeat(epoch, len(pos_nparr))])
    perm = np.random.permutation(len(seq_nparr))
    seq_nparr = np.array([seq_nparr[i] for i in perm])
    order_label = order_label[perm]
    dataset = to_dataloader(seq_nparr, label_nparr)
    return dataset, seq_nparr, order_label


def remove_old_seq(order_label, seq_nparr, num_to_add):
    to_remove = np.argsort(order_label)[:num_to_add]
    # print("@@@@@@@@@@", np.argsort(order_label), to_remove)
    seq_nparr = np.array(
        [d for i, d in enumerate(seq_nparr) if i not in to_remove])
    order_label = np.delete(order_label, to_remove)
    return seq_nparr, order_label
