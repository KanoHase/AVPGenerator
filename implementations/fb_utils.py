import os
import numpy as np
import time
import random
from FunctionAnalyser import TransClassifier
from implementations.data_utils import load_data_esm, to_dataloader
from esm_main.initialize_esm import gen_repr


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
    # bad_indices = (all_preds <= preds_cutoff).nonzero()[0]
    pos_seqs = [list(sampled_seqs[i]) for i in good_indices]
    # neg_seqs = [list(sampled_seqs[i]) for i in bad_indices]
    return pos_seqs


def meta_select_pos(sampled_seqs, epoch, preds_cutoff):
    make_input_file(sampled_seqs, epoch)

    preds = make_pred(epoch)
    good_indices = (preds > preds_cutoff).nonzero()[0]
    # bad_indices = (preds <= preds_cutoff).nonzero()[0]
    pos_seqs = [list(sampled_seqs[i]) for i in good_indices]
    # neg_seqs = [list(sampled_seqs[i]) for i in bad_indices]
    return pos_seqs


def make_input_file(sampled_seqs, epoch):
    input_path = './data_fbgan/input/'

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    fasta_seqs = [('>'+str(i)+'\n'+seq+'\n')
                  for (i, seq) in enumerate(sampled_seqs)]
    joint_fasta_seqs = ''.join(fasta_seqs)

    with open(input_path+'input_{0}.txt'.format(epoch), mode='w') as f:
        f.write(joint_fasta_seqs)


def make_pred(epoch):
    output_path = './data_fbgan/output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
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


def update_data_ps(pos_nparr, seq_nparr, order_label, label_nparr, epoch, data_size):
    if len(seq_nparr) == data_size:
        dataset, seq_nparr, order_label = update_data(
            pos_nparr, seq_nparr, order_label, label_nparr, epoch)
        return dataset, seq_nparr, order_label

    else:
        if data_size < len(pos_nparr) + len(seq_nparr):
            # just one epoch before the data reach up to data_size
            pos_num = data_size-len(seq_nparr)
            seq_nparr = np.concatenate([seq_nparr, pos_nparr[:pos_num]])
        elif len(pos_nparr) > 0:
            pos_num = len(pos_nparr)
            seq_nparr = np.concatenate([seq_nparr, pos_nparr])
        else:
            pos_num = len(pos_nparr)
            seq_nparr = seq_nparr
        order_label = np.concatenate(
            [order_label, np.repeat(epoch, pos_num)])
        perm = np.random.permutation(len(seq_nparr))
        seq_nparr = np.array([seq_nparr[i] for i in perm])
        order_label = order_label[perm]
        dataset = to_dataloader(seq_nparr, label_nparr)
        return dataset, seq_nparr, order_label


def remove_old_seq(order_label, seq_nparr, num_to_add):
    to_remove = np.argsort(order_label)[:num_to_add]
    seq_nparr = np.array(
        [d for i, d in enumerate(seq_nparr) if i not in to_remove])
    order_label = np.delete(order_label, to_remove)
    return seq_nparr, order_label


def soften_pos_seq(pos_seqs, rand_seqs, fbprop):
    pos_num = int(len(pos_seqs) * fbprop)
    rand_num = int(len(pos_seqs) - pos_num)
    soften_pos_seqs = random.sample(pos_seqs, pos_num)
    soften_rand_seqs = random.sample(rand_seqs, rand_num)
    mixed_pos_seq = soften_pos_seqs + soften_rand_seqs
    return mixed_pos_seq


def mutate_seqs(sampled_seqs, mutatepr, a_list):
    mutated_sampled_seqs = []
    total_len = sum(len(seq) for seq in sampled_seqs)
    flag_lis = [False, True]
    prob_lis = [1-mutatepr, mutatepr]
    rep_lis = np.random.choice(
        a=flag_lis, size=total_len, p=prob_lis)
    i = 0

    for seq in sampled_seqs:
        mutseq = ""
        for aa in seq:
            if rep_lis[i]:
                while True:
                    j = random.randint(0, len(a_list)-1)
                    if aa != a_list[j]:
                        break
                # replace amino
                mutseq += a_list[j]
            else:
                mutseq += aa
            i += 1

        mutated_sampled_seqs.append(mutseq)

    return mutated_sampled_seqs
