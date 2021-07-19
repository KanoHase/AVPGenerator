#!/usr/bin/zsh

utarr=("PR-PS")
fearr=(0.5 0.75 1)
fparr=(0.5)

for i in ${utarr[@]}
for j in ${fearr[@]}
for k in ${fparr[@]}
do
  python3 WGANgp_main.py --ut $i --fe $j --fp $k
done
