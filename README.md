# AVP Generator

## System Requirements and Installation
- Python 3.8.8

## Demo Instructions
All default arguments for demo are provided.
1)~3) can be completed with `zsh main.sh`

1) Run `python ./implementations/preprocess.py` to get ready for the dataset.
  OUT: `./real_data`, `./real_data_fasta` 

2) Run `python pretrain_classification.py` to pretrain the Transformer Function Analyser.
  OUT: `./figures`

3) Run `python WGANgp_main.py` to train WGAN with Gradient Penalty to produce valid gene sequences.
  OUT: `./samples/(parameter_combination)`, `./samples_fasta/(parameter_combination)`, `./figures/(parameter_combination)` 

4) Run alphafold2 for Pos, Neg and Syn. 
  OUT: `./alphafold_out` 

5) Run `python eval_metaiavp.py` to predict antiviral activities using Meta-iAVP.
  Run `python eval_metaiavp.py --pair` for pairwise alignment, it would take time.
  OUT: `./eval/(parameter_combination)/preds etc`, `./eval/val_positive/preds etc`, `./eval/val_negative/preds etc` , `./eval/preds.txt`

6) Run `python evaluate.py` to evaluate the generated sequences.
  OUT: `./eval/(parameter_combination)`, `./eval/val_positive`, `./eval/*png`, `./eval/preds, properties, stdev, all, ranking`


## Note
- Data alteration
  - ./raw_data/
    - positive.txt, line 34: Chandipura Virus → Chandipura virus
    - positive.txt, line 35: Chandipura virus , Vesicular stomatitis virus → Chandipura virus, Vesicular stomatitis virus
    - positive.txt, line 46: l → I
    - positive.txt, line 540: Paramyxovirus → paramyxoviruses
    - val_negative_exp.txt, 22: RGGRLCYARRRFAVCVGRb → RGGRLCYARRRFAVCVGRB