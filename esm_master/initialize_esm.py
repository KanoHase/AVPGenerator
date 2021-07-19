import torch
import esm
import numpy as np

#model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
# Downloading: "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt" to /home/kano_hasegawa/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S-contact-regression.pt


def gen_repr(data):
    # Load ESM-1b model
    # model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # batch_labels:['protein1', 'protein2'], batch_strs:['MK...G', 'KA...E'], batch_tokens:tensor([[ 0, 20, ..., 1],[0, 15, ... 2]]), batch_tokens.shape:torch.Size([2, 73])
    # 0:beginning sign, 1:padding, 2:end sign

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]
    token_representations = token_representations.numpy()
    # token_representations.shape:torch.Size([2, 73, 1280]), len(results):4, results.keys():['logits', 'representations', 'attentions', 'contacts']
    # results["logits"].shape:torch.Size([2, 73, 33]), results["attentions"]:torch.Size([2, 33, 20, 73, 73]), results["contacts"].shape:torch.Size([2, 71, 71])
    # 1280:Embedding Dim, 33:number of layers, 20:attention_head, 71:seq length

    # Generate per-sequence representations via averaging
    # note: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
    # sequence_representations:2×1280
    sequence_representations = np.array(sequence_representations)
    print("!!!!!!!!", sequence_representations.shape)

    return sequence_representations


"""
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
]

gen_repr(data)
"""
