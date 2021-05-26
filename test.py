from implementations.fb_utils import update_data
import numpy as np


def check_update_data():
    seq_nparr = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]])
    order_label = np.array([0, 0, 0, 0])
    label_nparr = np.array([0, 0, 0, 0])

    for epoch in range(1, 5):
        print("---------", seq_nparr, order_label)
        pos_seq = np.array([[0, 0, epoch+10], [0, 0, epoch+10]])
        dataset, seq_nparr, order_label = update_data(
            pos_seq, seq_nparr, order_label, label_nparr, epoch)
        print(seq_nparr, order_label)


check_update_data()
