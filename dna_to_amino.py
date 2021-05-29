from implementations.bio_utils import geneToProtein

aa_data_dir = "./data/"
sample_file = "dna_uniprot_under_50_reviewed.fasta"

dna_seqs = []

with open(aa_data_dir + sample_file) as f:
    for line in f:
        dna_seqs.append(line[:-1])

p_seqs = geneToProtein(dna_seqs)

with open(aa_data_dir + "random_seq.txt", "w") as f:
    for line in p_seqs:
        f.write(line)
        f.write("\n")
