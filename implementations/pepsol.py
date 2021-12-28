'''
- Overall soluble restrictions:
https://bioserv.rpbs.univ-paris-diderot.fr/services/SolyPep/index.html#PepBasic3D

- hydropho_aa_lis and gelprone_aa_lis:
https://www.thermofisher.com/jp/ja/home/life-science/protein-biology/protein-biology-learning-center/protein-biology-resource-library/pierce-protein-methods/peptide-design.html

'''

from modlamp.descriptors import GlobalDescriptor

charged_aa_lis = ['K', 'R', 'H', 'D', 'E']
hydropho_aa_lis = ['A', 'V', 'I', 'L', 'M', 'F', 'W']  # G, P, Y, C
gelprone_aa_lis = ['D', 'E', 'H', 'K', 'N', 'Q' 'R' 'S', 'T', 'Y']


def startend(seq):
    if (seq[0] in charged_aa_lis) or (seq[len(seq)-1] in charged_aa_lis):
        return False
    else:
        return True


def gorp(seq):
    limnum = len(seq)/5
    if seq.count("G") + seq.count("P") > limnum:
        return False
    else:
        return True


def aa_perseq(seq):
    limnum = len(seq)*0.25
    aas = set(seq)

    for aa in aas:
        if (seq.count(aa) > limnum):
            return False
    return True


def hydro_perseq(seq):
    limnum = len(seq)*0.45
    haasum = 0
    caasum = 0

    for aa in hydropho_aa_lis:
        haasum += seq.count(aa)
    for aa in charged_aa_lis:
        caasum += seq.count(aa)

    aasum = max(haasum, caasum)

    if aasum > limnum:
        return False
    return True


def gel_perseq(seq):
    limnum = len(seq)*0.75
    aasum = 0

    for aa in gelprone_aa_lis:
        aasum += seq.count(aa)

    if aasum > limnum:
        return False
    return True


def charge(seq):
    glob_seq = GlobalDescriptor(seq)
    glob_seq.calculate_charge()

    if glob_seq.descriptor[0][0] > 8:
        return False
    return True


def tf(seq):
    if not startend(seq):
        return False
    if not gorp(seq):
        return False
    if not aa_perseq(seq):
        return False
    if not hydro_perseq(seq):
        return False
    if not gel_perseq(seq):
        return False
    if not charge(seq):
        return False
    else:
        return True


def soluble_out(seq_list):
    sol_seq_dic = {}

    seq_set = set(seq_list)

    for seq in seq_set:
        flag = tf(seq)
        sol_seq_dic[seq] = flag

    return sol_seq_dic
