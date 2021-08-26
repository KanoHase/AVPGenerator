#!/usr/bin/zsh

utarr=("PR-PS" "P-PS" "R-S")
fearr=(0.25 0.5 0.75 1)
fparr=(0.25 0.5 0.75 1)
mparr=(0 0.1)
#revarr=("red-shuf-rep-revr" "red-shuf-rep" "red" "shuf-rep")

python3 WGANgp_main.py --ut "PR-PS" --fe 0
python3 WGANgp_main.py --ut "P-PS" --fe 0
python3 WGANgp_main.py --ut "R-S" --fe 0


for i in ${utarr[@]}
for j in ${fearr[@]}
for k in ${fparr[@]}
for l in ${mparr[@]}
#for m in ${revarr[@]}
do
  # python3 pretrain_classification.py --revise $m
  python3 WGANgp_main.py --ut $i --fe $j --fp $k --mp $l #--rev $m
done
