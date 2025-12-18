import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import numpy as np
import pydicom

# IMPORT CONFIG FROM SETUP FILE
from data_pipeline_setup import CLASS_MAPPING

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class UNIFESPXRayDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, class_mapping: dict, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.num_classes = len(class_mapping)
        self.id_to_idx = {id: idx for idx, id in enumerate(class_mapping.keys())}
        self.labels = self._prepare_labels()

    def _prepare_labels(self):
        labels_dict = {}
        for index, row in self.dataframe.iterrows():
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            targets_str = row['Target']
            if pd.notna(targets_str):
                try:
                    target_ids = [int(t) for t in targets_str.split(' ') if t.isdigit()]
                    for t_id in target_ids:
                        if t_id in self.id_to_idx: 
                            multi_hot[self.id_to_idx[t_id]] = 1.0
                except ValueError: pass
            labels_dict[index] = multi_hot
        return labels_dict

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['FilePath']
        
        # --- WINDOWS LONG PATH FIX ---
        # Convert to absolute path
        abs_path = os.path.abspath(img_path)
        
        # On Windows, prepend \\?\ to allow paths longer than 260 chars
        if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
            abs_path = '\\\\?\\' + abs_path
            
        try:
            # Read using the absolute long path
            dicom_data = pydicom.dcmread(abs_path)
            
            image_array = dicom_data.pixel_array
            if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                image_array = np.amax(image_array) - image_array
            
            image_array = image_array - np.min(image_array)
            if np.max(image_array) != 0:
                image_array = image_array / np.max(image_array)
            image_array = (image_array * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
        except Exception as e:
            # Only print error if it's NOT the "No such file" to avoid spamming console
            # during long training if a few files are truly missing
            if "No such file" not in str(e):
                print(f"Error loading {img_path}: {e}")
            image = Image.new('L', (224, 224), 0)
        
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx], row['SOPInstanceUID']

# Transforms
BASELINE_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])