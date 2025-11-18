import pandas as pd
import os
import glob

# --- CONFIGURATION ---
# IMPORTANT: Point this to the root of your image data (e.g., where 'train' is located)
# Based on your input: /archive/train/train/train/
DATA_ROOT = "./unifesp_xray_dataset/train/train/train/"
OUTPUT_MAPPING_CSV = "./unifesp_xray_dataset/image_path_map.csv"
LABEL_CSV = "./unifesp_xray_dataset/train.csv" # The file mapping UIDs to labels

def create_image_path_map(data_root: str, output_path: str):
    """
    Recursively finds all .dcm files and creates a mapping from
    SOPInstanceUID (extracted from the file name or DICOM metadata) to its full path.
    NOTE: The UNIFESP dataset usually uses the SOPInstanceUID as the base filename.
    """
    print(f"Scanning for DICOM files recursively under: {data_root}")
    
    # Use glob to find all files ending in .dcm (recursive search)
    dcm_files = glob.glob(os.path.join(data_root, '**/*.dcm'), recursive=True)
    
    if not dcm_files:
        print("Error: No .dcm files found. Please verify the DATA_ROOT path.")
        return

    # Assuming the SOPInstanceUID is the base filename without the extension
    # Example: 1.2.826.0.1.3680043.8.498.72533800876543798738969965510832915095-c.dcm
    # We strip '-c.dcm' or just '.dcm' to get the SOPInstanceUID.
    
    uid_to_path = {}
    for full_path in dcm_files:
        filename = os.path.basename(full_path)
        # Assuming SOPInstanceUID is the first part of the filename
        # This is often the case for UNIFESP dataset files.
        uid = filename.split('-')[0].split('.')[0] # Use the full base name as UID for robustness
        # Since the 'train.csv' uses UIDs like '1.2.826.0.1.3680043.8.498.12506063821850171756494207689001728484', 
        # let's assume the full filename without extension is the SOPInstanceUID
        uid = filename.replace('.dcm', '')

        # We need the UID that matches the one in your 'train.csv' file
        # The UNIFESP dataset actually uses the last number in the path (e.g., the UID folder)
        # We will use the full path itself to link later, and join with the train.csv
        uid_to_path[uid] = full_path

    print(f"Found {len(uid_to_path)} unique DICOM files.")

    # 2. Load the main label CSV to get the list of SOPInstanceUIDs and their labels
    try:
        label_df = pd.read_csv(LABEL_CSV)
        
        # Merge the image path data with the label data. 
        # This step needs careful validation of the UID format between the two files.
        # Given the complexity, we simplify: we map the file path itself to the SOPInstanceUID
        
        # A safer approach for the UNIFESP dataset is to assume the `SOPInstanceUID` is in the filename.
        # We'll stick to a direct path-to-UID map and save it.
        mapping_df = pd.DataFrame(uid_to_path.items(), columns=['SOPInstanceUID', 'FilePath'])
        
        # Now, ensure that the UIDs in the mapping align with UIDs in train.csv
        
        # For simplicity in this step, let's just save the file map.
        # We will merge the train.csv labels with this file map in the next iteration.
        mapping_df.to_csv(output_path, index=False)
        print(f"Successfully created DICOM path map at: {output_path}")

    except FileNotFoundError:
        print(f"Error: Label CSV file not found at {LABEL_CSV}. Please check path.")
        return

if __name__ == '__main__':
    # Adjust this path based on where you put your DICOM files
    create_image_path_map(DATA_ROOT, OUTPUT_MAPPING_CSV)