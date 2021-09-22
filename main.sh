#!/usr/bin/zsh

# python3 implementations/preprocess.py
# python3 pretrain_classification.py

eparr=(50 75 100 150)
baarr=(32 64 128)
lrarr=(0.00001 0.0001 0.001)
mparr=(0.7 0.8 0.9)
#revarr=("red-shuf-rep-revr" "red-shuf-rep" "red" "shuf-rep")

for i in ${eparr[@]}
for j in ${baarr[@]}
for k in ${lrarr[@]}
for l in ${mparr[@]}
#for m in ${revarr[@]}

do
  # python3 pretrain_classification.py --revise $m
  python3 WGANgp_main.py --epoch $i --batch $j --lr $k --preds_cutoff $l #--rev $m
done
