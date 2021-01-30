# AVP Generator

## System Requirements and Installation
- Python 3.6.3

## Demo Instructions
All default arguments for demo are provided.

1) Run `python WGANgp_main.py` to train WGAN with Gradient Penalty to produce valid gene sequences.
  - **Expected Output**
    - `sample/$RUN_NAME` will contain sample gene sequences from every 100 iterations, as well as loss and distance curves.
    - `amino_transformer_posneg.tsv` and `amino_word2vec_vectors.vec` will appear in `data` directory
