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
from unifesp_xray_dataset_handler import UNIFESPXRayDataset, BASELINE_TRANSFORMS

warnings.filterwarnings("ignore")

# CONFIG
NUM_CLASSES = len(CLASS_MAPPING)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10 # Increased epochs since scheduler helps convergence
MODEL_NAME = "ResNet-18-Gatekeeper-Advanced"

def load_data_splits():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_split.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_split.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_split.csv'))
    
    train_ds = UNIFESPXRayDataset(train_df, CLASS_MAPPING, transform=BASELINE_TRANSFORMS)
    val_ds = UNIFESPXRayDataset(val_df, CLASS_MAPPING, transform=BASELINE_TRANSFORMS)
    test_ds = UNIFESPXRayDataset(test_df, CLASS_MAPPING, transform=BASELINE_TRANSFORMS)
    
    # Calculate Class Weights for Imbalance Handling
    print("Calculating class weights...")
    class_counts = torch.zeros(NUM_CLASSES)
    for _, label, _ in tqdm(train_ds, desc="Scanning Dataset"):
        class_counts += label # Label is a multi-hot vector
    
    # pos_weight = (num_negatives) / (num_positives)
    # This forces the model to treat positive instances of rare classes as very important
    num_samples = len(train_ds)
    pos_weights = (num_samples - class_counts) / (class_counts + 1e-5) # Add epsilon to avoid div by zero
    pos_weights = pos_weights.to(DEVICE)
    
    print(f"Class Weights Calculated (Min: {pos_weights.min():.2f}, Max: {pos_weights.max():.2f})")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, pos_weights

def setup_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Unfreeze all layers for maximum performance
    for param in model.parameters():
        param.requires_grad = True
    
    num_ftrs = model.fc.in_features
    # REMOVED SIGMOID: We use BCEWithLogitsLoss which applies Sigmoid internally
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

def calculate_f1_score(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_pred_binary, average='macro', zero_division=0)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    all_targets, all_preds = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets, _ in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Apply Sigmoid manually here for evaluation metrics
            probs = torch.sigmoid(outputs)
            
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    f1 = calculate_f1_score(y_true, y_pred)
    return avg_loss, f1

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    print(f"\n--- Training {MODEL_NAME} on {DEVICE} ---")
    
    best_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, targets, _ in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        train_loss = running_loss / len(train_loader)
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion)
        
        # Update Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.2e}")
        
        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(DATA_DIR, 'resnet18_baseline_weights.pth'))
            print(f"  >>> Best Model Saved! ({best_f1:.4f})")

    return model

if __name__ == '__main__':
    train_loader, val_loader, test_loader, pos_weights = load_data_splits()
    
    model = setup_model(NUM_CLASSES)
    
    # WEIGHTED LOSS: Forces model to focus on rare classes
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # SCHEDULER: Reduce LR if validation loss stops dropping for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    # Final Test
    print("\n--- Running Final Test Evaluation ---")
    # Reload best weights
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'resnet18_baseline_weights.pth'), map_location=DEVICE, weights_only=False))
    test_loss, test_f1 = evaluate_model(model, test_loader, criterion)
    
    print("\n" + "="*40)
    print(f"UPGRADED MODEL RESULT (Test F1): {test_f1:.4f}")
    print("="*40)