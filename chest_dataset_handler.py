import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import numpy as np

# NIH Chest X-ray Dataset Mean/Std (different from ImageNet, but ImageNet is often used as a proxy)
# We will stick to ImageNet stats for transfer learning compatibility.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The 14 official pathologies in NIH Dataset + "No Finding"
ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

class NIHChestXrayDataset(Dataset):
    """
    Dataset loader for NIH Chest X-ray dataset (Multi-Label).
    Expects a CSV file with an 'Image Index' column and a 'Finding Labels' column.
    """
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.labels = ALL_LABELS
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = row['Image Index']
        
        # Handle full path if provided in CSV, else assume relative to img_dir
        if os.path.isabs(img_name):
            img_path = img_name
        else:
            img_path = os.path.join(self.img_dir, img_name)
            
        # 1. Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            # Create black placeholder if missing (robustness)
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 2. Process Labels (Multi-Hot Encoding)
        # Labels are strings like "Infiltration|Nodule"
        label_str = row['Finding Labels']
        label_vec = torch.zeros(len(self.labels), dtype=torch.float32)
        
        if label_str != "No Finding":
            for i, pathology in enumerate(self.labels):
                if pathology in label_str:
                    label_vec[i] = 1.0
        
        # 3. Apply Transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label_vec, img_name

# Standard Preprocessing for DenseNet-121
CHEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Augmentation for Training (Optional but recommended for Phase 2)
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])