# import os
# import glob
# import pandas as pd
# from natsort import natsorted

# # ================= CONFIGURATION =================
# # 1. Where are your Bias Corrected images located?
# SEARCH_DIR = "/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/OASIS_MRI_Session/BiasCorrected"

# # 2. Where do you want to save the CSV files?
# OUTPUT_CSV_DIR = "/home/biomedialab/Music/TrueNet_WMH/CSV_Lists"
# # =================================================

# def create_csvs():
#     if not os.path.exists(SEARCH_DIR):
#         print(f"Error: Directory not found: {SEARCH_DIR}")
#         return

#     os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
#     print(f"Scanning: {SEARCH_DIR} ...")

#     # 1. Find ALL .nii.gz files recursively
#     all_files = glob.glob(os.path.join(SEARCH_DIR, "**", "*.nii.gz"), recursive=True)
    
#     # 2. Separate them into lists
#     flair_paths = []
#     t1_paths = []

#     for filepath in all_files:
#         filename = os.path.basename(filepath)
        
#         # Check for FLAIR
#         if "FLAIR" in filename and "mask" not in filename:
#             flair_paths.append(filepath)
            
#         # Check for T1w
#         elif "T1w" in filename and "mask" not in filename:
#             t1_paths.append(filepath)

#     # 3. Sort them naturally (so sub-1, sub-2, sub-10 comes in correct order)
#     flair_paths = natsorted(flair_paths)
#     t1_paths = natsorted(t1_paths)

#     print(f"Found {len(flair_paths)} FLAIR images.")
#     print(f"Found {len(t1_paths)} T1w images.")

#     # 4. Save FLAIR CSV
#     if flair_paths:
#         df_flair = pd.DataFrame(flair_paths, columns=["FLAIR"])
#         save_path_flair = os.path.join(OUTPUT_CSV_DIR, "flair_images.csv")
#         df_flair.to_csv(save_path_flair, index=False)
#         print(f"Saved: {save_path_flair}")

#     # 5. Save T1 CSV
#     if t1_paths:
#         df_t1 = pd.DataFrame(t1_paths, columns=["T1"])
#         save_path_t1 = os.path.join(OUTPUT_CSV_DIR, "t1_images.csv")
#         df_t1.to_csv(save_path_t1, index=False)
#         print(f"Saved: {save_path_t1}")

#     print("Done!")

# if __name__ == "__main__":
#     create_csvs()


import os
import glob
import pandas as pd
from natsort import natsorted

# ================= CONFIGURATION =================
# Where are your SynthSeg-in-FLAIR images stored?
SEARCH_DIR = "/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/Linear_Registration/FLAIR_SPACE_OUTPUT"

# Where do you want to save the CSV?
OUTPUT_CSV_DIR = "/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/CSV_Lists"
# =================================================

def create_flair_synthseg_csv():
    if not os.path.exists(SEARCH_DIR):
        print(f"Error: Directory not found: {SEARCH_DIR}")
        return

    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    print(f"Scanning: {SEARCH_DIR} ...")

    # Find ONLY SynthSeg-in-FLAIR images
    synthseg_flair_paths = glob.glob(
        os.path.join(SEARCH_DIR, "**", "*T1w_synthseg_in_FLAIR.nii.gz"),
        recursive=True
    )

    # Natural sort
    synthseg_flair_paths = natsorted(synthseg_flair_paths)

    print(f"Found {len(synthseg_flair_paths)} SynthSeg-in-FLAIR images.")

    if not synthseg_flair_paths:
        print("No files found. Exiting.")
        return

    # Save CSV
    df = pd.DataFrame(synthseg_flair_paths, columns=["FLAIR"])
    save_path = os.path.join(OUTPUT_CSV_DIR, "flair_synthseg_images.csv")
    df.to_csv(save_path, index=False)

    print(f"Saved CSV: {save_path}")
    print("Done!")

if __name__ == "__main__":
    create_flair_synthseg_csv()
