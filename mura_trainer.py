import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import os
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np
import warnings

# Import your MURA handler
from mura_dataset_handler import MURADataset, TRAIN_TRANSFORMS, VAL_TRANSFORMS

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_DIR = "./mura_dataset/" 

# Hyperparameters
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5 # Increased for scheduler
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mura_model():
    print("Loading DenseNet-121 for MURA...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 1) # Linear output for BCEWithLogitsLoss
    return model.to(DEVICE)

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss() # Standard loss for evaluation metric calculation
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_preds)
    kappa = cohen_kappa_score(all_targets, all_preds)
    
    return avg_loss, accuracy, kappa

def train_mura():
    print(f"--- Starting Advanced Phase 2B Training (MURA) on {DEVICE} ---")
    
    # 1. Prepare Data
    train_dataset = MURADataset(DATA_DIR, split='train', transform=TRAIN_TRANSFORMS)
    val_dataset = MURADataset(DATA_DIR, split='valid', transform=VAL_TRANSFORMS)
    
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Check your DATA_DIR path.")
        return

    print(f"Dataset Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # --- WEIGHT CALCULATION FOR IMBALANCE ---
    print("Calculating class weights...")
    # Quick scan to count normals vs abnormals
    neg_count = 0
    pos_count = 0
    # Access underlying data list directly for speed instead of loading images
    for _, label in train_dataset.data:
        if label == 1: pos_count += 1
        else: neg_count += 1
            
    print(f"Stats: Normal (0): {neg_count}, Abnormal (1): {pos_count}")
    
    # Weight = Negatives / Positives
    pos_weight = torch.tensor([neg_count / (pos_count + 1e-5)]).to(DEVICE)
    print(f"Calculated pos_weight: {pos_weight.item():.4f}")

    # 2. Setup Model
    model = get_mura_model()
    
    # Weighted Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Reduce LR when Validation Kappa stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, verbose=True)
    
    best_kappa = 0.0
    best_save_path = "mura_densenet121_best.pth"
    
    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        val_loss, val_acc, val_kappa = evaluate_model(model, val_loader)
        
        # Step Scheduler based on Kappa (Maximize Kappa)
        scheduler.step(val_kappa)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val Kappa: {val_kappa:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")
        
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), best_save_path)
            print(f"  >>> New Best Model Saved! (Kappa: {best_kappa:.4f})")

    final_path = "mura_densenet121_final.pth"
    torch.save(model.state_dict(), final_path)
    print("Training Complete.")

if __name__ == "__main__":
    train_mura()