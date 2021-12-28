# Evolutionary Scale Modeling


This repository contains code and pre-trained weights for **Transformer protein language models** from Facebook AI Research, including our state-of-the-art **ESM-1b** and **MSA Transformer**.
Transformer protein language models were introduced in our paper, ["Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences" (Rives et al., 2019)](https://doi.org/10.1101/622803).

**ESM-1b outperforms all tested single-sequence protein language models across a range of structure prediction tasks.**
The MSA Transformer (ESM-MSA-1) can improve performance further by leveraging MSA information.

<details><summary>Citation</summary>

```bibtex
@article{rives2019biological,
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
  title={Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences},
  year={2019},
  doi={10.1101/622803},
  url={https://www.biorxiv.org/content/10.1101/622803v4},
  journal={bioRxiv}
}
```
</details>

<details><summary>Table of contents</summary>
  
- [Main models you should use](#main-models)
- [Comparison to related works](#perf_related)
- [Usage](#usage)
  - [Quick Start](#quickstart)
  - [Compute embeddings in bulk from FASTA](#bulk_fasta)
  - [Zero-shot variant prediction](#zs_variant)
  - [Notebooks](#notebooks)
- [Available Models and Datasets](#available)
  - [Pre-trained Models](#available-models)
  - [ESM Structural Split Dataset](#available-esmssd)
  - [Pre-training Dataset Split](#available-pretraining-split)
- [Citations](#citations)
- [License](#license)
</details>

<details><summary>What's New</summary>
  
- August 2021: Added flexibility to tokenizer to allow for spaces and special tokens (like `<mask>`) in sequence.
- July 2021: New pre-trained model ESM-1v released, trained on UniRef90 (see [Meier et al. 2021](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1)).
- July 2021: New MSA Transformer released, with a minor fix in the row positional embeddings (`ESM-MSA-1b`).
- Feb 2021: MSA Transformer added (see [Rao et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1)). Example usage in [notebook](#notebooks).
- Dec 2020: [Self-Attention Contacts](#notebooks) for all pre-trained models (see [Rao et al. 2020](https://doi.org/10.1101/2020.12.15.422761))
- Dec 2020: Added new pre-trained model [ESM-1b](#perf_related) (see [Rives et al. 2019](https://doi.org/10.1101/622803) Appendix B)
- Dec 2020: [ESM Structural Split Dataset](#available-esmssd) (see [Rives et al. 2019](https://doi.org/10.1101/622803) Appendix A.10)
  
</details>

## Main models you should use <a name="main-models"></a>

| Shorthand | `esm.pretrained.`           | Dataset | Description  |
|-----------|-----------------------------|---------|--------------|
| ESM-1b    | `esm1b_t33_650M_UR50S()`       | UR50  | SOTA general-purpose protein language model. Can be used to predict structure, function and other protein properties directly from individual sequences. Released with [Rives et al. 2019](https://doi.org/10.1101/622803) (Dec 2020 update). |
| ESM-MSA-1b| `esm_msa1b_t12_100M_UR50S()` |  UR50 + MSA  | MSA Transformer language model. Can be used to extract embeddings from an MSA. Enables SOTA inference of structure. Released with [Rao et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v2) (ICML'21 version, June 2021).  |
| ESM-1v    | `esm1v_t33_650M_UR90S_1()` ... `esm1v_t33_650M_UR90S_5()`| UR90  | Language model specialized for prediction of variant effects. Enables SOTA zero-shot prediction of the functional effects of sequence variations. Same architecture as ESM-1b, but trained on UniRef90. Released with [Meier et al. 2021](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1). |

For a complete list of available models, with details and release notes, see [Pre-trained Models](#available-models).

## Comparison to related works <a name="perf_related"></a>
<!--
DO NOT EDIT THIS TABLE! This is the master copy:
https://docs.google.com/spreadsheets/d/1RPvWF47rIMEr-Jg-SRCoGElHcwCl5d7RyEeSyPgp59A/edit#gid=0
exported via https://www.tablesgenerator.com/html_tables
-->



<table>
<thead>
  <tr>
    <th>Task</th>
    <th colspan="3">Unsupervised contact prediction</th>
    <th colspan="2">Supervised contact prediction</th>
    <th>SSP</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Test set</td>
    <td>Large valid</td>
    <td>CASP13-FM</td>
    <td>CAMEO</td>
    <td>CASP13-FM</td>
    <td>CAMEO</td>
    <td>CB513</td>
  </tr>
  <tr>
    <td>Gremlin (Potts)</td>
    <td>39.3</td>
    <td>16.9</td>
    <td>24.0</td>
    <td>40.1</td>
    <td>47.3</td>
    <td></td>
  </tr>
  <tr>
    <td>UniRep</td>
    <td></td>
    <td></td>
    <td></td>
    <td>11.2</td>
    <td>17.8</td>
    <td>58.4</td>
  </tr>
  <tr>
    <td>SeqVec</td>
    <td></td>
    <td></td>
    <td></td>
    <td>13.8</td>
    <td>22.5</td>
    <td>62.1</td>
  </tr>
  <tr>
    <td>TAPE</td>
    <td>11.2</td>
    <td>5.5</td>
    <td>6.8</td>
    <td>12.3</td>
    <td>15.9</td>
    <td>58.0</td>
  </tr>
  <tr>
    <td>ProtBert-BFD</td>
    <td>34.1</td>
    <td>13.5</td>
    <td>23.9</td>
    <td>24.7</td>
    <td>37.0</td>
    <td>70.0</td>
  </tr>
  <tr>
    <td>Prot-T5-XL-BFD</td>
    <td>35.6</td>
    <td>16.5</td>
    <td>25.9</td>
    <td>25.0</td>
    <td>40.8</td>
    <td>71.4 ± 0.3</td>
  </tr>
  <tr>
    <td>ESM-1</td>
    <td>33.7</td>
    <td>13.6</td>
    <td>21.4</td>
    <td>(todo)</td>
    <td>(todo)</td>
    <td>69.2</td>
  </tr>
  <tr>
    <td>ESM-1b</td>
    <td>41.1</td>
    <td>17.0</td>
    <td>30.9</td>
    <td>28.2</td>
    <td>44.4</td>
    <td>71.6 ± 0.1</td>
  </tr>
  <tr>
    <td>ESM-1v</td>
    <td>35.3</td>
    <td>14.2</td>
    <td>24.4</td>
    <td> </td>
    <td> </td>
    <td> </td>
  </tr>
  <tr>
    <td>ESM-MSA-1b</td>
    <td>57.4</td>
    <td>44.8</td>
    <td>43.5</td>
    <td>54.6</td>
    <td>55.8</td>
    <td>73.4 ± 0.3</td>
  </tr>
</tbody>
</table>

Comparison to related protein language models on structure prediction tasks.

* All contact numbers are the top-L,LR precision metric, where long range means sequence separation of at least 24 residues
* For unsupervised contact prediction, a sparse linear combination of the attention heads is used to directly predict protein contacts,
fitted with logistic regression on 20 structures. 
For more details on the method, see [Rao et al. 2020](https://doi.org/10.1101/2020.12.15.422761).
* Supervised contact prediction all uses the same resnet (32 layers) and trRosetta training data, cf [Rao et al. 2021](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v2).
* (SSP) Secondary structure Q8 accuracy on CB513, transformer finetuned with convolution + LSTM head.
* Direct coupling analysis methods (Gremlin, mfDCA, Psicov) and ESM-MSA-1 use the [trRosetta MSAs](https://yanglab.nankai.edu.cn/trRosetta/benchmark/), while other methods predict from single sequence.


## Usage <a name="usage"></a>

### Quick Start <a name="quickstart"></a>

As a prerequisite, you must have PyTorch 1.5 or later installed to use this repository.

You can use this one-liner for installation:

```bash
$ pip install fair-esm
```

We also support PyTorch Hub, which removes the need to clone and/or install this repository yourself:

```python
import torch
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
```

Then, you can load and use a pretrained model as follows:

```python
import torch
import esm

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "K A <mask> I S Q"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, (_, seq) in enumerate(data):
    sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))

# Look at the unsupervised self-attention map contact predictions
import matplotlib.pyplot as plt
for (_, seq), attention_contacts in zip(data, results["contacts"]):
    plt.matshow(attention_contacts[: len(seq), : len(seq)])
    plt.title(seq)
    plt.show()
```

### Compute embeddings in bulk from FASTA <a name="bulk_fasta"></a>

We provide a script that efficiently extracts embeddings in bulk from a FASTA file.
A cuda device is optional and will be auto-detected.
The following command extracts the final-layer embedding for a FASTA file from the ESM-1b model:

```bash
$ python extract.py esm1b_t33_650M_UR50S examples/some_proteins.fasta examples/some_proteins_emb_esm1b/ \
    --repr_layers 0 32 33 --include mean per_tok
```

Directory `examples/some_proteins_emb_esm1b/` now contains one `.pt` file per FASTA sequence; use `torch.load()` to load them.
`extract.py` has flags that determine what's included in the `.pt` file:
* `--repr-layers` (default: final only) selects which layers to include embeddings from.
* `--include` specifies what embeddings to save. You can use the following:
  * `per_tok` includes the full sequence, with an embedding per amino acid (seq_len x hidden_dim).
  * `mean` includes the embeddings averaged over the full sequence, per layer.
  * `bos` includes the embeddings from the beginning-of-sequence token. 
  (NOTE: Don't use with the pre-trained models - we trained without bos-token supervision)

### Zero-shot variant prediction <a name="zs_variant"></a>
See "[./variant-prediction/](variant-prediction/)" for code and pre-trained weights for the ESM-1v models described in 
[Language models enable zero-shot prediction of the effects of mutations on protein function. (Meier et al. 2021)](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1).


### Notebooks <a name="notebooks"></a> 

#### Supervised variant prediction - training a classifier on the embeddings

[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/facebookresearch/esm/blob/master/examples/sup_variant_prediction.ipynb)


To help you get started with using the embeddings, this [jupyter notebook tutorial](examples/sup_variant_prediction.ipynb) shows how to train a supervised variant predictor using embeddings from ESM-1.
You can adopt a similar protocol to train a model for any downstream task, even with limited data.
First you can obtain the embeddings for ``examples/P62593.fasta`` either by [downloading the precomputed](https://dl.fbaipublicfiles.com/fair-esm/examples/P62593_reprs.tar.gz) embeddings
as instructed in the notebook or by running the following:

```bash
# Obtain the embeddings
$ python extract.py esm1_t34_670M_UR50S examples/P62593.fasta examples/P62593_reprs/ \
    --repr_layers 34 --include mean
```

Then, follow the remaining instructions in the tutorial. You can also run the tutorial in a [colab notebook](https://colab.research.google.com/github/facebookresearch/esm/blob/master/examples/variant_prediction.ipynb).

**Note this is somewhat outdated: use `esm1v_t33_650M_UR90S` instead, and
see [the newer instructions for zero-shot variant prediction](variant-prediction/),
that is without any supervised training.**


#### Unsupervised contact prediction
[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb)

This [jupyter notebook tutorial](examples/contact_prediction.ipynb) demonstrates contact prediction with both the ESM-1b and MSA Transformer (ESM-MSA-1) models.
Contact prediction is based on a logistic regression over the model's attention maps.
This methodology is based on our ICLR 2021 paper, 
[Transformer protein language models are unsupervised structure learners. (Rao et al. 2020)](https://doi.org/10.1101/2020.12.15.422761)
The MSA Transformer (ESM-MSA-1) takes a multiple sequence alignment (MSA) as input, and uses the tied row self-attention maps in the same way.
See [MSA Transformer. (Rao et al. 2021)](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1).

To get unsupervised attention-based contacts, call `model.predict_contacts(tokens)` or `model(tokens, return_contacts=True)`.


#### ESMStructuralSplitDataset and self-attention contact prediction
[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/facebookresearch/esm/blob/master/examples/esm_structural_dataset.ipynb)

And this [jupyter notebook tutorial](examples/esm_structural_dataset.ipynb) shows how to load and index the `ESMStructuralSplitDataset`,
and computes the self-attention map unsupervised contact predictions using ESM-1b.


## Available Models and Datasets <a name="available"></a>

### Pre-trained Models <a name="available-models"></a>

| Shorthand | `esm.pretrained.`           | #layers | #params | Dataset | Embedding Dim |  Model URL (automatically downloaded to `~/.cache/torch/hub/checkpoints`) |
|-----------|---------------------|---------|---------|---------|---------------|-----------------------------------------------------------------------|
| ESM-1v    | `esm1v_t33_650M_UR90S_[1-5]` | 33     | 650M    | UR90/S 2020_03  | 1280          | https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt   |
| ESM-MSA-1b| `esm_msa1b_t12_100M_UR50S` | 12     | 100M    | UR50/S + MSA 2018_03 | 768        | https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt   |
| ESM-MSA-1 | `esm_msa1_t12_100M_UR50S` | 12     | 100M    | UR50/S + MSA 2018_03 | 768        | https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1_t12_100M_UR50S.pt   |
| ESM-1b    | `esm1b_t33_650M_UR50S` | 33     | 650M    | UR50/S 2018_03 | 1280          | https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt   |
| ESM-1     | `esm1_t34_670M_UR50S` | 34      | 670M    | UR50/S 2018_03 | 1280          |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50S.pt |
|           | `esm1_t34_670M_UR50D` | 34      | 670M    | UR50/D 2018_03 | 1280          |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50D.pt |
|           | `esm1_t34_670M_UR100` | 34      | 670M    | UR100 2018_03  | 1280          |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR100.pt |
|           | `esm1_t12_85M_UR50S`  | 12      | 85M     | UR50/S 2018_03 | 768           |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt  |
|           | `esm1_t6_43M_UR50S`   | 6       | 43M     | UR50/S 2018_03 | 768           |  https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t6_43M_UR50S.pt   |


Here is a chronological list of the released models and the paper they were introduced in:

| Shorthand | Release Notes |
|-----------|---------------|
| ESM-1     | Released with Rives et al. 2019 (Aug 2020 update). |
| ESM-1b    | Released with Rives et al. 2019 (Dec 2020 update). See Appendix B. |
| ESM-MSA-1 | Released with Rao et al. 2021 (Preprint v1). |
| ESM-MSA-1b | Released with Rao et al. 2021 (ICML'21 version, June 2021). |
| ESM-1v     | Released with Meier et al. 2021. |

### ESM Structural Split Dataset <a name="available-esmssd"></a>
This is a five-fold cross validation dataset of protein domain structures that can be used to measure generalization of representations
across different levels of structural dissimilarity.
The dataset implements structural holdouts at the family, superfamily, and fold
level. The SCOPe database is used to classify domains. Independently for each level of structural hold-out,
the domains are split into 5 equal sets, i.e. five sets of folds, superfamilies, or families. This ensures
that for each of the five partitions, structures having the same classification do not appear in both the
train and test sets. For a given classification level each structure appears in a test set once, so that
in the cross validation experiment each of the structures will be evaluated exactly once.

The dataset provides 3d coordinates, distance maps, and secondary structure labels.
For further details on the construction of the dataset 
see [Rives et al. 2019](https://doi.org/10.1101/622803) Appendix A.10.

This [jupyter notebook tutorial](examples/esm_structural_dataset.ipynb) shows how to load and index the `ESMStructuralSplitDataset`.

`ESMStructuralSplitDataset`, upon initializing, will download `splits` and `pkl`. 
We also provide `msas` for each of the domains. The data can be directly downloaded below.

| Name   | Description                                                                   | URL                                                                   |
|--------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| splits | train/valid splits                                                            | https://dl.fbaipublicfiles.com/fair-esm/structural-data/splits.tar.gz |
| pkl    | pkl objects containing sequence, SSP labels, distance map, and 3d coordinates | https://dl.fbaipublicfiles.com/fair-esm/structural-data/pkl.tar.gz    |
| msas   | a3m files containing MSA for each domain                                      | https://dl.fbaipublicfiles.com/fair-esm/structural-data/msas.tar.gz   | 

### Pre-training Dataset Split  <a name="available-pretraining-split"></a>
The split files establishing which UniRef50 clusters were used as held-out evaluation set for pre-training
in [Rives et al. 2019](https://doi.org/10.1101/622803) and [Rao et al. 2021](https://doi.org/10.1101/2021.02.12.430858) can be found here:
* [UniRef50 IDs of evaluation set](https://dl.fbaipublicfiles.com/fair-esm/pretraining-data/uniref201803_ur50_valid_headers.txt.gz): 3.016 M clusters
* [UniRef100 IDs of evaluation set](https://dl.fbaipublicfiles.com/fair-esm/pretraining-data/uniref201803_ur100_valid_headers.txt.gz): 13.745 M proteins, expanding the same UniRef50 clusters.

These files only contain only the UniRef50 IDs and UniRef100 IDs corresponding to the [UniRef database, 2018-03 release](https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2018_03/uniref/)
which is released by the UniProt Consortium under a [Creative Commons Attribution (CC BY 4.0) License](https://www.uniprot.org/help/license).

## Citations <a name="citations"></a>

If you find the models useful in your research, we ask that you cite the relevant paper:

```bibtex
@article{rives2019biological,
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
  title={Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences},
  year={2019},
  doi={10.1101/622803},
  url={https://www.biorxiv.org/content/10.1101/622803v4},
  journal={bioRxiv}
}
```
 
For the self-attention contact prediction:

```bibtex
@article{rao2020transformer,
  author = {Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov, Sergey and Rives, Alexander},
  title={Transformer protein language models are unsupervised structure learners},
  year={2020},
  doi={10.1101/2020.12.15.422761},
  url={https://www.biorxiv.org/content/10.1101/2020.12.15.422761v1},
  journal={bioRxiv}
}
```

For the MSA Transformer:

```bibtex
@article{rao2021msa,
  author = {Rao, Roshan and Liu, Jason and Verkuil, Robert and Meier, Joshua and Canny, John F. and Abbeel, Pieter and Sercu, Tom and Rives, Alexander},
  title={MSA Transformer},
  year={2021},
  doi={10.1101/2021.02.12.430858},
  url={https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1},
  journal={bioRxiv}
}
```

For variant prediction using ESM-1v:

```bibtex
@article{meier2021language,
  author = {Meier, Joshua and Rao, Roshan and Verkuil, Robert and Liu, Jason and Sercu, Tom and Rives, Alexander},
  title = {Language models enable zero-shot prediction of the effects of mutations on protein function},
  year={2021},
  doi={10.1101/2021.07.09.450648},
  url={https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1},
  journal={bioRxiv}
}
```

Much of this code builds on the [fairseq](https://github.com/pytorch/fairseq) sequence modeling framework. We use fairseq internally for our protein language modeling research. We highly recommend trying it out if you'd like to pre-train protein language models from scratch.

## License <a name="license"></a>

This source code is licensed under the MIT license found in the `LICENSE` file
in the root directory of this source tree.
