import os
import glob
import re
import numpy as np
import nibabel as nib
import pandas as pd

# ================= CONFIGURATION =================
# Path where your FLAIR segmentation maps are stored
INPUT_DIR = "/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/WHM_Seg_Maps_T1w/"

# Where to save the final CSV
OUTPUT_CSV = "/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/Suresh/TrueNet_WMH/wmh_volumes_T1w.csv"
# =================================================

def get_voxel_volume(img):
    """Calculates the volume of a single voxel in mm^3."""
    zooms = img.header.get_zooms()
    return np.prod(zooms[:3])

def extract_info(filename):
    """
    Extracts Subject ID (clean) and Visit Time from filenames.
    Format example: Predicted_probmap_truenet_sub-OAS30001_sess-d4467.nii.gz
    """
    # 1. Find Subject ID 
    # CHANGED HERE: 'sub-' is outside the parenthesis, so we only capture 'OAS30001'
    sub_match = re.search(r'sub-([a-zA-Z0-9]+)', filename)
    subject = sub_match.group(1) if sub_match else "Unknown"
    
    # 2. Find Visit Time (digits immediately following 'd')
    visit_match = re.search(r'd(\d+)', filename)
    
    if visit_match:
        visit_time = int(visit_match.group(1)) # Convert to integer
    else:
        visit_time = 0 
    return subject, visit_time

def main():
    results = []
    
    print(f"Searching for masks in: {INPUT_DIR}")
    search_pattern = os.path.join(INPUT_DIR, "**", "*.nii.gz")
    mask_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(mask_files)} files. Calculating volumes...")

    for filepath in mask_files:
        filename = os.path.basename(filepath)
        
        try:
            # Load the image
            img = nib.load(filepath)
            data = img.get_fdata()
            
            # Calculate Volume
            voxel_vol_mm3 = get_voxel_volume(img)
            
            # Count lesion pixels (usually value >= 0.5 for probability maps)
            lesion_pixel_count = np.sum(data >= 0.45)
            
            total_vol_mm3 = lesion_pixel_count * voxel_vol_mm3
            total_vol_ml = total_vol_mm3 / 1000.0
            
            # Extract Subject and Visit Time
            subject, visit_time = extract_info(filename)
            
            results.append({
                "Subject_ID": subject,
                "Visit_Time_Days": visit_time,
                "Volume_mm3": round(total_vol_mm3, 2),
                "Volume_mL": round(total_vol_ml, 4)
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Create DataFrame
    df = pd.DataFrame(results)
    
    if not df.empty:
        # Sort by Subject first, then by Visit Time
        df = df.sort_values(by=["Subject_ID", "Visit_Time_Days"])
        
        # Save to CSV
        df.to_csv(OUTPUT_CSV, index=False)
        
        print("------------------------------------------------")
        print(f"Successfully processed {len(df)} masks.")
        print(f"Results saved to: {OUTPUT_CSV}")
        print("------------------------------------------------")
        print(df.head())
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()




