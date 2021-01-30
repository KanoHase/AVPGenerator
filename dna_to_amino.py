from implementations.bio_utils import geneToProtein

aa_data_dir = "./data/"
sample_file = "random_dna_seqs.fa"

dna_seqs = []

with open(aa_data_dir + sample_file) as f:
    for line in f:
        dna_seqs.append(line[:-1])

p_seqs,valid_gene_seqs = geneToProtein(dna_seqs)

with open (aa_data_dir + "amino_" + sample_file, "w") as f:
    for line in p_seqs:
        f.write(line)
        f.write("\n") 
