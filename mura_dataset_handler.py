import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

# MURA uses ImageNet stats for transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class MURADataset(Dataset):
    """
    Dataset loader for MURA (Musculoskeletal Radiographs).
    Crawls the directory structure to find images and labels.
    MURA is binary: 0 (Normal) vs 1 (Abnormal).
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split # 'train' or 'valid'
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        # MURA directory structure:
        # root/train/XR_ELBOW/patientX/studyY_positive/image.png
        
        image_paths = []
        labels = []
        
        # Handle cases where user might extract MURA-v1.1 inside the folder
        target_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(target_dir):
            target_dir = os.path.join(self.root_dir, "MURA-v1.1", self.split)
            
        if not os.path.exists(target_dir):
            # Fallback: Just return empty if not found, let the user fix path
            print(f"Warning: Could not find '{self.split}' folder in {self.root_dir}")
            return []

        print(f"Scanning MURA {self.split} data...")
        
        # Walk through body parts (XR_ELBOW, XR_FINGER, etc.)
        for body_part in os.listdir(target_dir):
            body_part_dir = os.path.join(target_dir, body_part)
            if not os.path.isdir(body_part_dir): continue

            # Walk through patients
            for patient in os.listdir(body_part_dir):
                patient_dir = os.path.join(body_part_dir, patient)
                if not os.path.isdir(patient_dir): continue

                # Walk through studies (positive vs negative)
                for study in os.listdir(patient_dir):
                    study_dir = os.path.join(patient_dir, study)
                    if not os.path.isdir(study_dir): continue

                    # Determine label from folder name
                    # study1_positive -> 1 (Abnormal)
                    # study1_negative -> 0 (Normal)
                    is_abnormal = 1 if 'positive' in study else 0
                    
                    for img in os.listdir(study_dir):
                        if img.endswith('.png') or img.endswith('.jpg'):
                            image_paths.append(os.path.join(study_dir, img))
                            labels.append(is_abnormal)
                            
        print(f"Found {len(image_paths)} images for {self.split}.")
        return list(zip(image_paths, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        # Return float tensor for BCE Loss
        return image, torch.tensor(label, dtype=torch.float32)

# Data Augmentation is crucial for MURA to prevent overfitting
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])