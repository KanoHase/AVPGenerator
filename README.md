# AVP Generator

## System Requirements and Installation
- Python 3.8.8

## Demo Instructions
All default arguments for demo are provided.

1) Run `python ./implementations/preprocess.py` to get ready for the dataset.
  e.g. `python ./implementations/preprocess.py --screen --vir_min 1 --noexp`

2) Run `python pretrain_classification.py` to pretrain the Transformer Function Analyser.
  Remember to comment out the different binary_negative_data_file and binary_negative_val_data_file
  e.g. if 1) is `python ./implementations/preprocess.py --screen --vir_min 1 --noexp`, comment out "negative_exp.txt" and "val_negative_exp.txt" row
  - **Expected Output**

3) Run `python WGANgp_main.py` or `zsh main.sh`to train WGAN with Gradient Penalty to produce valid gene sequences.

3) Run `python evaluate.py` to evaluate the generated sequences.

## Note
- Data alteration
  - ./raw_data/
    - positive.txt, 34: Chandipura Virus → Chandipura virus
    - positive.txt, 35: Chandipura virus , Vesicular stomatitis virus → Chandipura virus, Vesicular stomatitis virus
    - positive.txt, line 46: l → I
    - positive.txt, 540: Paramyxovirus → paramyxoviruses

  - ./subsamples/data/
    - positive.txt, line 133: J → L
    - positive.txt, line 475: Delete O
    - positive.txt, line 177: Delete data
    - positive.txt, line 497: X → A
    - negative_noexp.txt, line 303: Delete data
    - negative_noexp.txt, line 350: Z → E
    - negative_noexp.txt, line 351: B → D
    - val_negative_noexp.txt, line 14: X → A
    - val_negative_noexp.txt, line 28: B → D (4 times)
    - val_negative_noexp.txt, line 28: Z → E