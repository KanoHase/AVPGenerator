#!/usr/bin/zsh
python3 implementations/preprocess.py
python3 pretrain_classification.py

fearr=(0 1)

for i in ${fearr[@]}

do
  python3 WGANgp_main.py --fe $i
done
