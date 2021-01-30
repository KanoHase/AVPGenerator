from implementations.bio_utils import geneToProtein
from statistics import mean, variance

sample_dir = "./samples/fbgan_avp_demo_1of4_400/"
else_sample_dir = "./samples/else/"
withoutfb_sample_dir = "./samples/without_fb_fbgan_avp_demo/"
aa_data_dir = "./data/"
eval_dir = "./eval/"
length_dir = "/length/"
molw_dir = "/MW/"
pi_dir = "/PI/"
sample_file = "sampled_400_preds.txt"
else_sample_file = "thr_0.8.txt"
withoutfb_sample_file = "sampled_400_preds.txt"
# realdata_file = "amino_positive_544.txt"
realdata_file = "amino_eval_positive_60.txt"
neg_realdata_file = "amino_negative_544.txt"
aacomp_file = "aacomp.txt"
summary_file = "summary.txt"
models = ["newFBGAN", "FBGAN", "real", "neg"]

### non-auto: /PI, length.txt

def length(p_seqs, else_p_seqs, real_seqs, neg_seqs):
    print("Length")
    lens = []
    else_lens = []
    real_lens = []
    neg_lens = []

    for seq in p_seqs:
        lens.append(len(seq))
    print("FBGAN_new:\t", mean(lens), "\t", variance(lens))

    for seq in else_p_seqs:
        else_lens.append(len(seq))
    print("FBGAN:\t", mean(else_lens), "\t", variance(else_lens))

    for seq in real_seqs:
        real_lens.append(len(seq))
    print("Real:\t", mean(real_lens), "\t", variance(real_lens), "\n")

    for seq in neg_seqs:
        neg_lens.append(len(seq))
    print("Negative:\t", mean(neg_lens), "\t", variance(neg_lens), "\n")

    return lens, else_lens, real_lens, neg_lens

def molw(p_seqs, else_p_seqs, real_seqs, neg_seqs):
    print("MolecularWeight")
    weight_table = {
    'G':57.051, 'A':71.078, 'S':87.077, 'P':97.115,
    'V':99.131, 'T':101.104, 'C':103.143, 'I':113.158,
    'L':113.158, 'N':114.103, 'D':115.087, 'Q':128.129,
    'K':128.172, 'E':129.114, 'M':131.196, 'H':137.139,
    'F':147.174, 'R':156.186, 'Y':163.173, 'W':186.210,
    'O':255.31, #pyrrolysine
    'B':114.595, #D or N
    'Z':128.622, #E or Q
}

    mw = []
    else_mw = []
    real_mw = []
    neg_mw = []

    for seq in p_seqs:
        w = 0
        for aa in seq:
            if aa == "X":
                w = 0
                break
            w += weight_table[aa]
        if w != 0:
            mw.append(w)
    print("FBGAN_new:\t", mean(mw), "\t", variance(mw))

    for seq in else_p_seqs:
        w = 0
        for aa in seq:
            if aa == "X":
                w = 0
                break
            w += weight_table[aa]
        if w != 0:
            else_mw.append(w)
    print("FBGAN:\t", mean(else_mw), "\t", variance(else_mw))

    for seq in real_seqs:
        w = 0
        for aa in seq:
            if aa == "X":
                # print("!!!!")
                w = 0
                break
            w += weight_table[aa]
        if w != 0:
            real_mw.append(w)
    print("Real:\t", mean(real_mw), "\t", variance(real_mw), "\n")

    for seq in neg_seqs:
        w = 0
        for aa in seq:
            if aa == "X":
                # print("!!!!")
                w = 0
                break
            w += weight_table[aa]
        if w != 0:
            neg_mw.append(w)
    print("Negative:\t", mean(neg_mw), "\t", variance(neg_mw), "\n")

    return mw, else_mw, real_mw, neg_mw

def aroma(p_seqs, else_p_seqs, real_seqs, neg_seqs):
    print("Aromaticity")

    aromatic = ['F', 'W', 'H', 'Y']

    size = 0
    arom = 0
    for seq in p_seqs:
        for aa in seq:
            size += 1
            if aa in aromatic:
                arom += 1
    aromaticity = arom/size
    print("FBGAN_new:\t", aromaticity)

    size = 0
    arom = 0
    for seq in else_p_seqs:
        for aa in seq:
            size += 1
            if aa in aromatic:
                arom += 1
    else_aromaticity = arom/size
    print("FBGAN:\t", else_aromaticity)

    size = 0
    arom = 0
    for seq in real_seqs:
        for aa in seq:
            size += 1
            if aa in aromatic:
                arom += 1
    real_aromaticity = arom/size
    print("Real:\t", real_aromaticity, "\n")

    size = 0
    arom = 0
    for seq in neg_seqs:
        for aa in seq:
            size += 1
            if aa in aromatic:
                arom += 1
    neg_aromaticity = arom/size
    print("Negative:\t", neg_aromaticity, "\n")

    return aromaticity, else_aromaticity, real_aromaticity, neg_aromaticity



def aacomp(p_seqs, else_p_seqs, real_seqs, neg_seqs):
    print("Amino Acid Composition")

    glob_aa_freq = {
    'G':0, 'A':0, 'S':0, 'P':0,
    'V':0, 'T':0, 'C':0, 'I':0,
    'L':0, 'N':0, 'D':0, 'Q':0,
    'K':0, 'E':0, 'M':0, 'H':0,
    'F':0, 'R':0, 'Y':0, 'W':0,
    'O':0, 'X':0, #pyrrolysine
    'B':114.595, #D or N
    'Z':128.622, #E or Q
    }

    aa_freq = glob_aa_freq.copy() #値渡ししないといけないので
    size = 0
    for seq in p_seqs:
        for aa in seq:
            size += 1
            aa_freq[aa] += 1

    for key in aa_freq:
        aa_freq[key] = aa_freq[key]/size
    print("FBGAN_new:\t", aa_freq, "\t")

    else_aa_freq = glob_aa_freq.copy()
    size = 0
    for seq in else_p_seqs:
        for aa in seq:
            size += 1
            else_aa_freq[aa] += 1
    for key in else_aa_freq:
        else_aa_freq[key] = else_aa_freq[key]/size
    print("FBGAN:\t", else_aa_freq, "\t")

    real_aa_freq = glob_aa_freq.copy()
    size = 0
    for seq in real_seqs:
        for aa in seq:
            size += 1
            real_aa_freq[aa] += 1
    for key in real_aa_freq:
        real_aa_freq[key] = real_aa_freq[key]/size
    print("Real:\t", real_aa_freq, "\n")

    neg_aa_freq = glob_aa_freq.copy()
    size = 0
    for seq in neg_seqs:
        for aa in seq:
            size += 1
            neg_aa_freq[aa] += 1
    for key in neg_aa_freq:
        neg_aa_freq[key] = neg_aa_freq[key]/size
    print("Negative:\t", neg_aa_freq, "\n")

    print(real_aa_freq, neg_aa_freq)

    return aa_freq, else_aa_freq, real_aa_freq, neg_aa_freq


def pi():

    for i, model in enumerate(models):
        pis = []
        with open(eval_dir + pi_dir + "pi_" + model + ".txt") as f:
            for line in f:
                pi_num = float(line[:-1])
                pis.append(pi_num)

            if i == 0:
                pi_list = pis
            if i == 1:
                else_pi_list = pis
            if i == 2:
                real_pi_list = pis
            if i == 3:
                neg_pi_list = pis

    return pi_list, else_pi_list, real_pi_list, neg_pi_list


def main():
    #################################preparation#################################
    dna_seqs = []
    else_dna_seqs = []
    withoutfb_dna_seqs = []

    with open(sample_dir + sample_file) as f:
        for line in f:
            dna_seqs.append(line[:-1])

    p_seqs,valid_gene_seqs = geneToProtein(dna_seqs)
    # print(p_seqs, len(p_seqs), len(dna_seqs))

    with open(else_sample_dir + else_sample_file) as f:
        for line in f:
            else_dna_seqs.append(line[:-1])

    else_p_seqs,valid_gene_seqs = geneToProtein(else_dna_seqs)

    with open(withoutfb_sample_dir + withoutfb_sample_file) as f:
        for line in f:
            withoutfb_dna_seqs.append(line[:-1])

    withoutfb_p_seqs,valid_gene_seqs = geneToProtein(withoutfb_dna_seqs)

    real_seqs = []
    with open(aa_data_dir + realdata_file) as f:
        for line in f:
            real_seqs.append(line[:-1])

    neg_seqs = []
    with open(aa_data_dir + neg_realdata_file) as f:
        for line in f:
            neg_seqs.append(line[:-1])

    #################################calculation#################################
    ###length
    lens, else_lens, real_lens, neg_lens = length(p_seqs, else_p_seqs, real_seqs, neg_seqs)
    len_lists = [lens, else_lens, real_lens, neg_lens]
    # all_len = [[]]*max(len(lens), len(else_lens), len(real_lens))
    
    # for i in range(len(else_lens)):
    #     all_len[i].append(else_lens[i])
    #     break
    # print(all_len)

    for i, model in enumerate(models):
        with open(eval_dir + length_dir + model + ".txt", "w") as f:
            for l in len_lists[i]:
                f.write(str(l/len(len_lists[i])))
                f.write("\n")

    ###mw
    mw, else_mw, real_mw, neg_mw = molw(p_seqs, else_p_seqs, real_seqs, neg_seqs)
    mw_lists = [mw, else_mw, real_mw, neg_mw]

    for i, model in enumerate(models):
        with open(eval_dir + molw_dir + model + ".txt", "w") as f:
            for l in mw_lists[i]:
                f.write(str(l/len(mw_lists[i])))
                f.write("\n")

    ###aroma
    aromaticity, else_aromaticity, real_aromaticity, neg_aromaticity = aroma(p_seqs, else_p_seqs, real_seqs, neg_seqs)

    ###aacomp
    aa_freq, else_aa_freq, real_aa_freq, neg_aa_freq = aacomp(p_seqs, else_p_seqs, real_seqs, neg_seqs)

    with open(eval_dir + aacomp_file, "w") as f:
        f.write("\tFBGAN_new\tFBGAN\tReal\tNegative\tdif_FBGAN_new\tdif_FBGAN\tdif_neg\n")
        for key in aa_freq:
            list_tmp = [key, str(aa_freq[key]), str(else_aa_freq[key]), str(real_aa_freq[key]), str(neg_aa_freq[key]), str(abs(real_aa_freq[key]-aa_freq[key])),str(abs(real_aa_freq[key]-else_aa_freq[key])), str(abs(real_aa_freq[key]-neg_aa_freq[key])), "\n"]
            f.write("\t".join(list_tmp))

    ###pi
    pi_list, else_pi_list, real_pi_list, neg_pi_list = pi()

    
    #################################summary#################################
    with open(eval_dir + summary_file, "w") as f:
        f.write("\tLength\t\tMolecularWeight\t\tAromaticity\tPI\n\tMean\tVariance\tMean\tVariance\tMean\tMean\n")
        list_newFBGAN = ["new_FBGAN", str(mean(lens)), str(variance(lens)), str(mean(mw)),str(variance(mw)),str(aromaticity), str(mean(pi_list)), "\n"]
        f.write("\t".join(list_newFBGAN))
        list_FBGAN = ["FBGAN", str(mean(else_lens)), str(variance(else_lens)), str(mean(else_mw)),str(variance(else_mw)),str(else_aromaticity), str(mean(else_pi_list)), "\n"]
        f.write("\t".join(list_FBGAN))
        list_real = ["Real", str(mean(real_lens)), str(variance(real_lens)), str(mean(real_mw)),str(variance(real_mw)),str(real_aromaticity), str(mean(real_pi_list)), "\n"]
        f.write("\t".join(list_real))
        list_neg = ["Negative", str(mean(neg_lens)), str(variance(neg_lens)), str(mean(neg_mw)),str(variance(neg_mw)),str(neg_aromaticity), str(mean(neg_pi_list)), "\n"]
        f.write("\t".join(list_neg))


    #################################output amino acid#################################
    with open (aa_data_dir + "amino_" + sample_file, "w") as f:
        for line in p_seqs:
            f.write(line)
            f.write("\n") 

    with open (aa_data_dir + "amino_else_" + else_sample_file, "w") as f:
        for line in else_p_seqs:
            f.write(line)
            f.write("\n") 

    with open (aa_data_dir + "amino_withoutfb_" + withoutfb_sample_file, "w") as f:
        for line in withoutfb_p_seqs:
            f.write(line)
            f.write("\n") 

if __name__ == '__main__':
    main()
