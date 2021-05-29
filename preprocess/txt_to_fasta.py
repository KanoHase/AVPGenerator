from ..implementations.translator import text2fasta

data_dir = "./data/"
fasta_dir = "./fasta_data/"
real_pos_file = "positive_540"
real_pos_val_file = "val_positive_60"
real_neg_file = "negative_540"
real_neg_val_file = "val_negative_60"

file_list = [[data_dir, real_pos_file], [data_dir, real_pos_val_file], [
    data_dir, real_neg_file], [data_dir, real_neg_val_file]]

text2fasta(file_list, fasta_dir)
