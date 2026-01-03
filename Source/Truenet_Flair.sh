#!/bin/bash

# ================= CONFIGURATION =================
# 1. Path to your CSV file
csv_file="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/CSV_Lists/flair_synthseg_images.csv"

# 2. Where to save outputs
base_output_dir="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/WMH_Flair_SyntheSeg/"

# 3. Path to your pretrained model
model_path="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/ukbb_flair/"

# 4. Define Python and TrueNet paths explicitly
PYTHON_EXEC="/home/biomedialab/fsl/bin/python"
TRUENET_SCRIPT="/home/biomedialab/fsl/bin/truenet"
# =================================================

mkdir -p "$base_output_dir"

# Read the CSV file line by line (skipping the first header line)
tail -n +2 "$csv_file" | while read -r flair_image_path; do

    # 1. Skip empty lines
    if [ -z "$flair_image_path" ]; then continue; fi

    # 2. Check if the file actually exists
    if [ ! -f "$flair_image_path" ]; then
        echo "WARNING: File not found: $flair_image_path"
        continue
    fi

    echo "Processing: $flair_image_path"

    # 3. Get the DIRECTORY (Truenet needs the folder, not the file)
    input_dir=$(dirname "$flair_image_path")
    
    # 4. Get Subject ID for output folder
    filename=$(basename "$flair_image_path")
    # Extracts ID assuming format like sub-OAS30178_... -> sub-OAS30178
    subject_id=${filename%%_*} 
    
    subject_out_dir="${base_output_dir}/${subject_id}"
    mkdir -p "$subject_out_dir"

    # 5. Run TRUENET
    # We call the python executable directly on the script path
    "$PYTHON_EXEC" "$TRUENET_SCRIPT" evaluate \
        -i "$input_dir" \
        -o "$subject_out_dir" \
        -m "$model_path" \

done