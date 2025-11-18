import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import glob

# --- CENTRAL CONFIGURATION ---
# Adjust these paths to match your local setup
DATA_DIR = "./unifesp_xray_dataset/" 
LABEL_CSV = os.path.join(DATA_DIR, 'train.csv') 
# The root folder where your nested DICOM folders start
IMAGE_ROOT_DIR = os.path.join(DATA_DIR, 'train/train/train/') 

SEED = 42
TEST_SIZE = 0.20 
VALIDATION_SIZE = 0.20 

# UNIFESP Labels Mapping
CLASS_MAPPING = {
    0: 'Abdomen', 1: 'Ankle', 2: 'Cervical Spine', 3: 'Chest', 
    4: 'Clavicles', 5: 'Elbow', 6: 'Feet', 7: 'Finger', 
    8: 'Forearm', 9: 'Hand', 10: 'Hip', 11: 'Knee', 
    12: 'Lower Leg', 13: 'Lumbar Spine', 14: 'Others', 15: 'Pelvis', 
    16: 'Shoulder', 17: 'Sinus', 18: 'Skull', 19: 'Thigh', 
    20: 'Thoracic Spine'
}
NUM_CLASSES = len(CLASS_MAPPING)

def create_path_map(image_root: str):
    """Recursively finds all .dcm files and maps UID -> FilePath."""
    print(f"Scanning for DICOM files recursively under: {image_root}")
    dcm_files = glob.glob(os.path.join(image_root, '**/*.dcm'), recursive=True)
    
    uid_to_path = {}
    for full_path in dcm_files:
        filename = os.path.basename(full_path)
        # Clean the filename to get the UID
        uid = filename.replace('.dcm', '') 
        if uid.endswith('-c'): uid = uid.replace('-c', '')
        if uid.endswith('-d'): uid = uid.replace('-d', '')

        uid_to_path[uid] = full_path

    print(f"Found {len(uid_to_path)} unique DICOM files.")
    return pd.DataFrame(uid_to_path.items(), columns=['SOPInstanceUID', 'FilePath'])

def load_data_and_split(label_csv: str, image_root: str, test_size: float, seed: int):
    """Loads labels, merges with paths, and splits data."""
    label_df = pd.read_csv(label_csv)
    path_map_df = create_path_map(image_root)
    
    # Merge labels with file paths
    df_merged = pd.merge(label_df, path_map_df, on='SOPInstanceUID', how='inner')
    df_merged = df_merged.dropna(subset=['Target', 'FilePath'])
    
    print(f"Total merged samples: {len(df_merged)}")
    
    if len(df_merged) == 0:
        raise ValueError("Merge failed: Zero samples matched. Check CSV UIDs vs Filenames.")

    # Split based on unique IDs
    unique_ids = df_merged['SOPInstanceUID'].unique()
    train_ids, temp_ids = train_test_split(unique_ids, test_size=test_size, random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed)

    train_df = df_merged[df_merged['SOPInstanceUID'].isin(train_ids)].reset_index(drop=True)
    val_df = df_merged[df_merged['SOPInstanceUID'].isin(val_ids)].reset_index(drop=True)
    test_df = df_merged[df_merged['SOPInstanceUID'].isin(test_ids)].reset_index(drop=True)

    print(f"Split Sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

if __name__ == '__main__':
    try:
        train_df, val_df, test_df = load_data_and_split(LABEL_CSV, IMAGE_ROOT_DIR, TEST_SIZE, SEED)
        
        # Save splits to CSV
        train_df.to_csv(os.path.join(DATA_DIR, 'train_split.csv'), index=False)
        val_df.to_csv(os.path.join(DATA_DIR, 'val_split.csv'), index=False)
        test_df.to_csv(os.path.join(DATA_DIR, 'test_split.csv'), index=False)
        print("\nSUCCESS: Split CSVs created.")
        
    except Exception as e:
        print(f"Error: {e}")