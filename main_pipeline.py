import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import warnings

# Add the report generator import
# Ensure report_generator.py is in the same folder
try:
    from report_generator import RadiologyReportGenerator
except ImportError:
    RadiologyReportGenerator = None
    print("Warning: report_generator.py not found. Reporting features disabled.")

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Paths
# Update: Pointing to the specific folder where baseline_trainer saved the weights
GATEKEEPER_PATH = os.path.join("unifesp_xray_dataset", "resnet18_baseline_weights.pth")
CHEST_MODEL_PATH = "chest_densenet121_best.pth" # or _model.pth
MURA_MODEL_PATH = "mura_densenet121_best.pth"   # or _final.pth

# Class Mappings
BODY_PARTS = {
    0: 'Abdomen', 1: 'Ankle', 2: 'Cervical Spine', 3: 'Chest', 
    4: 'Clavicles', 5: 'Elbow', 6: 'Feet', 7: 'Finger', 
    8: 'Forearm', 9: 'Hand', 10: 'Hip', 11: 'Knee', 
    12: 'Lower Leg', 13: 'Lumbar Spine', 14: 'Others', 15: 'Pelvis', 
    16: 'Shoulder', 17: 'Sinus', 18: 'Skull', 19: 'Thigh', 
    20: 'Thoracic Spine'
}

CHEST_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# Standard Transforms (ImageNet Stats)
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultiDomainSystem:
    def __init__(self):
        print(f"Initializing Multi-Domain System on {DEVICE}...")
        self.gatekeeper = self._load_gatekeeper()
        self.chest_model = self._load_chest_model()
        self.mura_model = self._load_mura_model()
        print("All models loaded successfully.\n")

    def _load_gatekeeper(self):
        print(f"Loading Phase 1: Gatekeeper (ResNet-18) from {GATEKEEPER_PATH}...")
        model = models.resnet18(weights=None)
        
        # --- FIX: Match training architecture (Linear Only) ---
        # The saved weights are from a Linear layer, not a Sequential(Linear, Sigmoid)
        model.fc = nn.Linear(model.fc.in_features, len(BODY_PARTS))
        
        if os.path.exists(GATEKEEPER_PATH):
            model.load_state_dict(torch.load(GATEKEEPER_PATH, map_location=DEVICE, weights_only=False))
        else:
            print(f"Warning: {GATEKEEPER_PATH} not found. Gatekeeper will be random. check your unifesp_xray_dataset folder.")
        return model.to(DEVICE).eval()

    def _load_chest_model(self):
        print("Loading Phase 2A: Chest Specialist (DenseNet-121)...")
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(CHEST_LABELS))
        if os.path.exists(CHEST_MODEL_PATH):
            model.load_state_dict(torch.load(CHEST_MODEL_PATH, map_location=DEVICE, weights_only=False))
        else:
            print(f"Warning: {CHEST_MODEL_PATH} not found.")
        return model.to(DEVICE).eval()

    def _load_mura_model(self):
        print("Loading Phase 2B: Bone Specialist (DenseNet-121)...")
        model = models.densenet121(weights=None)
        # Binary classification (1 output)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        if os.path.exists(MURA_MODEL_PATH):
            model.load_state_dict(torch.load(MURA_MODEL_PATH, map_location=DEVICE, weights_only=False))
        else:
            print(f"Warning: {MURA_MODEL_PATH} not found.")
        return model.to(DEVICE).eval()

    def predict(self, img_path):
        # 1. Preprocess
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
        except Exception as e:
            return {"Error": f"Error loading image: {e}"}

        # 2. Run Gatekeeper
        with torch.no_grad():
            gk_out = self.gatekeeper(input_tensor)
            # Apply Sigmoid here during inference since we removed it from the model definition
            gk_probs = torch.sigmoid(gk_out)[0]
            top_prob, top_idx = torch.max(gk_probs, dim=0)
            body_part = BODY_PARTS[int(top_idx.item())]

        result = {
            "Step 1 (Classification)": f"{body_part} ({top_prob:.1%})"
        }

        # 3. Routing Logic
        if body_part == "Chest":
            # Route to Chest Specialist
            with torch.no_grad():
                chest_out = self.chest_model(input_tensor)
                chest_probs = torch.sigmoid(chest_out)[0]
            
            # Get findings > 5% confidence
            findings = []
            for i, prob in enumerate(chest_probs):
                if prob > 0.05: # Low threshold to see what it thinks
                    findings.append(f"{CHEST_LABELS[i]}: {prob:.1%}")
            
            if not findings: findings = ["No significant findings."]
            result["Step 2 (Deep Analysis)"] = "Chest Model Activated"
            result["Findings"] = findings

        elif body_part in ['Elbow', 'Finger', 'Forearm', 'Hand', 'Humerus', 'Shoulder', 'Wrist']:
            # Route to Bone Specialist (MURA)
            with torch.no_grad():
                mura_out = self.mura_model(input_tensor)
                mura_prob = torch.sigmoid(mura_out).item()
            
            status = "ABNORMAL" if mura_prob > 0.5 else "NORMAL"
            result["Step 2 (Deep Analysis)"] = "Bone/MURA Model Activated"
            result["Diagnosis"] = f"{status} (Abnormality Probability: {mura_prob:.1%})"

        else:
            # Anatomy not supported by specialists
            result["Step 2 (Deep Analysis)"] = "Skipped"
            result["Reason"] = f"No specialist model available for {body_part}."

        return result

if __name__ == "__main__":
    system = MultiDomainSystem()
    
    # Initialize Reporter if available
    reporter = RadiologyReportGenerator() if RadiologyReportGenerator else None
    
    print("="*50)
    print("Multi-Domain AI System Ready.")
    print("Type an image path to analyze (or 'q' to quit).")
    
    while True:
        user_input = input("\nImage Path >> ").strip()
        if user_input.lower() == 'q':
            break
            
        # Handle quoted paths (common in Windows copy-paste)
        user_input = user_input.strip('"').strip("'")
        
        if not os.path.exists(user_input):
            print("File not found.")
            continue
            
        # 1. Get Technical Analysis (JSON-like)
        analysis = system.predict(user_input)
        
        # 2. Generate Human-Readable Report
        if reporter and "Error" not in analysis:
            full_report = reporter.generate_report(analysis)
            print("\n" + full_report)
        else:
            # Fallback if reporter missing or error
            print("\n--- REPORT ---")
            for k, v in analysis.items():
                print(f"{k}: {v}")
            print("--------------")