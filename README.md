# AVP Generator

## System Requirements and Installation
- Python 3.8.8

## Demo Instructions
All default arguments for demo are provided.

1) Run `python ./implementations/preprocess.py` to get ready for the dataset.
  e.g. `python ./implementations/preprocess.py --screen --vir_min 1 --noexp`

2) Run `python pretrain_classification.py` to pretrain the Transformer Function Analyser.

3) Run `python WGANgp_main.py` or `zsh main.sh`to train WGAN with Gradient Penalty to produce valid gene sequences.

4) Run `python evaluate.py` to evaluate the generated sequences.

5) After deciding the optimal mode, run `python pairwise.py` to do pairwise alignment.

## Note
- Data alteration
  - ./raw_data/
    - positive.txt, 34: Chandipura Virus → Chandipura virus
    - positive.txt, 35: Chandipura virus , Vesicular stomatitis virus → Chandipura virus, Vesicular stomatitis virus
    - positive.txt, line 46: l → I
    - positive.txt, 540: Paramyxovirus → paramyxoviruses