import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import warnings

# IMPORTS FROM OTHER FILES
from data_pipeline_setup import CLASS_MAPPING, DATA_DIR
from dataset_handler import UNIFESPXRayDataset, BASELINE_TRANSFORMS

warnings.filterwarnings("ignore")

# CONFIG
NUM_CLASSES = len(CLASS_MAPPING)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 5
MODEL_NAME = "ResNet-18-Baseline"

def load_data_splits():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_split.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_split.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_split.csv'))
    
    train_ds = UNIFESPXRayDataset(train_df, CLASS_MAPPING, transform=BASELINE_TRANSFORMS)
    val_ds = UNIFESPXRayDataset(val_df, CLASS_MAPPING, transform=BASELINE_TRANSFORMS)
    test_ds = UNIFESPXRayDataset(test_df, CLASS_MAPPING, transform=BASELINE_TRANSFORMS)
    
    # Reduce num_workers to 0 to prevent multiprocessing errors during debugging
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def setup_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
    return model.to(DEVICE)

def calculate_f1_score(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_pred_binary, average='macro', zero_division=0)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for inputs, targets, _ in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
    
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return calculate_f1_score(y_true, y_pred)

def train_model(model, train_loader, val_loader, criterion, optimizer):
    print(f"\n--- Training {MODEL_NAME} on {DEVICE} ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, targets, _ in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        
        val_f1 = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch+1} Val F1 (Macro): {val_f1:.4f}")
    return model

if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data_splits()
    model = setup_model(NUM_CLASSES)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer)
    
    test_f1 = evaluate_model(trained_model, test_loader, criterion)
    print("\n" + "="*40)
    print(f"M2 BASELINE RESULT (Test F1): {test_f1:.4f}")
    print("="*40)