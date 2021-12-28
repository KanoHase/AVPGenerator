#!/usr/bin/zsh
#python3 implementations/preprocess.py
#python3 pretrain_classification.py

fearr=(0 0.75 1)

for i in ${fearr[@]}

do
  # python3 pretrain_classification.py --revise $m
  python3 WGANgp_main.py --fe $i
done
