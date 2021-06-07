# AVP Generator

## System Requirements and Installation
- Python 3.6.3

## Demo Instructions
All default arguments for demo are provided.

1) Create directories 
  - `data_fbgan/input/`
  - `data_fbgan/output/`

2) Run `python WGANgp_main.py` to train WGAN with Gradient Penalty to produce valid gene sequences.
  - **Expected Output**

## Note
- Data alteration
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