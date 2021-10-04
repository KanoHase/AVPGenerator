#!/usr/bin/zsh
python3 implementations/preprocess.py
#python3 pretrain_classification.py

siarr=(7 10 15)
oparr=('Adam' 'RAdam')
# eparr=(50 75 100 150)
# baarr=(32 64 128)
# lrarr=(0.00001 0.0001 0.001)
# pcarr=(0.7 0.8 0.9)
#revarr=("red-shuf-rep-revr" "red-shuf-rep" "red" "shuf-rep")

for i in ${siarr[@]}
for j in ${oparr[@]}
# for k in ${lrarr[@]}
# for l in ${pcarr[@]}
#for m in ${revarr[@]}

do
  # python3 pretrain_classification.py --revise $m
  python3 WGANgp_main.py --sample_itr $i --optimizer $j #--epoch $i --batch $j --lr $k --preds_cutoff $l --rev $m
done
