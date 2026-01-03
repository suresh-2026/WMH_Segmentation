# #!/bin/bash

# # ================= CONFIGURATION =================
# csv_file="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/CSV_Lists/t1_images.csv"
# base_output_dir="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/WHM_Seg_Maps_T1w/"
# model_path="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/ukbb_t1/"

# PYTHON_EXEC="/home/biomedialab/fsl/bin/python"
# TRUENET_SCRIPT="/home/biomedialab/fsl/bin/truenet"
# # =================================================

# mkdir -p "$base_output_dir"

# tail -n +2 "$csv_file" | while read -r t1_image_path; do

#     # 1. Skip empty lines and non-existent files
#     if [ -z "$t1_image_path" ]; then continue; fi
#     if [ ! -f "$t1_image_path" ]; then
#         echo "WARNING: File not found: $t1_image_path"
#         continue
#     fi

#     echo "Processing: $t1_image_path"

#     # 2. Extract IDs
#     filename=$(basename "$t1_image_path")
#     subject_id=${filename%%_*} 
    
#     # Get the filename WITHOUT .nii.gz to use as a unique base
#     # e.g., "sub-OAS30001_ses-d0129_T1w"
#     file_stem=$(basename "$t1_image_path" .nii.gz)

#     subject_out_dir="${base_output_dir}/${subject_id}"
#     mkdir -p "$subject_out_dir"

#     # 3. Create Temp Directory
#     temp_input_dir="${subject_out_dir}/temp_input_truenet"
#     mkdir -p "$temp_input_dir"

#     # ================= CHANGED HERE =================
#     # We use the unique 'file_stem' so the link name is unique.
#     # e.g., "sub-OAS30001_ses-d0129_T1w_T1.nii.gz"
#     # TrueNet will see the "_T1.nii.gz" at the end and be happy.
#     ln -sf "$t1_image_path" "${temp_input_dir}/${file_stem}_T1.nii.gz"
#     # ================================================

#     # 4. Run TrueNet
#     "$PYTHON_EXEC" "$TRUENET_SCRIPT" evaluate \
#         -i "$temp_input_dir" \
#         -o "$subject_out_dir" \
#         -m "$model_path" \
#         # -v

#     # 5. Clean up
#     rm -rf "$temp_input_dir"

# done



#!/bin/bash

# ================= USER INPUT =================
T1_IMAGE="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/bet_corrected_mri.nii.gz"
OUTPUT_DIR="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/Single_T1_Output"
MODEL_PATH="/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/ukbb_t1/"
# =============================================

PYTHON_EXEC="/home/biomedialab/fsl/bin/python"
TRUENET_SCRIPT="/home/biomedialab/fsl/bin/truenet"

mkdir -p "$OUTPUT_DIR"

# ================= SAFETY CHECK =================
if [ ! -f "$T1_IMAGE" ]; then
    echo "ERROR: T1 image not found!"
    exit 1
fi

# ================= PREP =================
filename=$(basename "$T1_IMAGE" .nii.gz)
temp_input_dir="${OUTPUT_DIR}/temp_input_truenet"

mkdir -p "$temp_input_dir"

# TrueNet REQUIRES "_T1.nii.gz"
ln -sf "$T1_IMAGE" "${temp_input_dir}/${filename}_T1.nii.gz"

# ================= RUN TRUENET =================
echo "Running TrueNet on single T1 image..."
"$PYTHON_EXEC" "$TRUENET_SCRIPT" evaluate \
    -i "$temp_input_dir" \
    -o "$OUTPUT_DIR" \
    -m "$MODEL_PATH"

# ================= CLEANUP =================
rm -rf "$temp_input_dir"

echo "✅ Done. Output saved in:"
echo "$OUTPUT_DIR"



