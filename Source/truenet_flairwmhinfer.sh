#!/bin/bash
export TRUENET_PRETRAINED_MODEL_PATH="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/ukbb_flair/"
for ((i=1; i<=1; i++)); do
    num=$(printf "%04d" $i)
    pre_flair_path="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/CSV_Lists/flair_images.csv"
    out_path="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/WMH_Seg_Maps/"
    echo "FLAIR path ${pre_flair_path}"
    truenet apply -i "$pre_flair_path" -o  "$out_path"  -m ukbb_flair
done
