import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings

# Import your handler
from chest_dataset_handler import NIHChestXrayDataset, CHEST_TRANSFORMS, ALL_LABELS

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_DIR = "./nih_chest_dataset/" 
CSV_FILE = os.path.join(DATA_DIR, "sample_labels.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")

# Hyperparameters
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 # Increased for scheduler
NUM_CLASSES = len(ALL_LABELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_densenet_model(num_classes):
    print("Loading DenseNet-121...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

def evaluate_model(model, loader):
    model.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs)
            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())
    
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    
    try:
        auc_score = roc_auc_score(val_targets, val_preds, average="macro")
    except ValueError:
        auc_score = 0.0
        
    return auc_score

def train_phase2():
    print(f"--- Starting Advanced Phase 2 Training (Chest) on {DEVICE} ---")
    
    # 1. Prepare Data
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found.")
        return

    full_dataset = NIHChestXrayDataset(df, IMG_DIR, transform=CHEST_TRANSFORMS)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # --- WEIGHTED LOSS CALCULATION ---
    print("Calculating class weights for imbalance handling...")
    # Iterate once to count positives (can be slow on huge datasets, but fine for sample)
    pos_counts = torch.zeros(NUM_CLASSES)
    # We use a temporary loader for speed
    temp_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    for _, labels, _ in tqdm(temp_loader, desc="Scanning Class Distribution"):
        pos_counts += labels.sum(dim=0)
        
    num_samples = len(train_dataset)
    # Weight = (Negatives / Positives)
    pos_weights = (num_samples - pos_counts) / (pos_counts + 1e-5)
    pos_weights = pos_weights.to(DEVICE)
    print(f"Weights calculated. Min: {pos_weights.min():.2f}, Max: {pos_weights.max():.2f}")

    # 2. Setup Model & Optimization
    model = get_densenet_model(NUM_CLASSES)
    
    # Weighted Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, verbose=True)
    
    best_val_auc = 0.0
    best_save_path = "chest_densenet121_best.pth"
    
    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for imgs, labels, _ in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        val_auc = evaluate_model(model, val_loader)
        
        # Step Scheduler based on AUC (We want to maximize AUC)
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.2e}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_save_path)
            print(f"  >>> New Best Model Saved! (AUC: {best_val_auc:.4f})")

    final_save_path = "chest_densenet121_final.pth"
    torch.save(model.state_dict(), final_save_path)
    print(f"Training Complete. Models saved.")

if __name__ == "__main__":
    train_phase2()