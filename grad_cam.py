import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import random

# Import configuration from your trainer/handler
from chest_dataset_handler import ALL_LABELS, IMAGENET_MEAN, IMAGENET_STD

# --- CONFIGURATION ---
DATA_DIR = "./nih_chest_dataset/"
IMG_DIR = os.path.join(DATA_DIR, "images")

# Smart Model Loading
if os.path.exists("chest_densenet121_best.pth"):
    MODEL_PATH = "chest_densenet121_best.pth"
    print(f"Configuration: Using BEST model checkpoint ({MODEL_PATH})")
else:
    MODEL_PATH = "chest_densenet121_model.pth"
    print(f"Configuration: Using standard model checkpoint ({MODEL_PATH})")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_for_inference(model_path):
    print(f"Loading model from {model_path}...")
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(ALL_LABELS))
    
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        self.hooks = []
        self.hooks.append(self.model.features.register_forward_hook(self.save_fmaps))
        self.hooks.append(self.model.features.register_full_backward_hook(self.save_grads))

    def save_fmaps(self, module, input, output):
        self.feature_maps = output.detach() if isinstance(output, torch.Tensor) else output

    def save_grads(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class_idx):
        features = self.model.features(input_tensor)
        
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        output = self.model.classifier(out)
        
        self.model.zero_grad()
        
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class_idx] = 1
        
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.feature_maps.clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        # Normalize 0-1 (Avoid division by zero)
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        
        return heatmap.cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(image).unsqueeze(0), image

def visualize(img_path, model):
    input_tensor, original_img = preprocess_image(img_path)
    input_tensor = input_tensor.to(DEVICE)
    
    # 1. Get Predictions
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0]
    
    top_prob, top_idx = torch.max(probs, dim=0)
    disease_name = ALL_LABELS[top_idx]
    
    print(f"Top Prediction: {disease_name} ({top_prob:.4f})")
    
    if top_prob < 0.01:
        print("Model is not confident (Prob < 0.01). Skipping CAM.")
        return

    # 2. Run Grad-CAM
    grad_cam = GradCAM(model)
    heatmap = grad_cam.generate_cam(input_tensor, top_idx)
    grad_cam.remove_hooks()
    
    # --- VISUALIZATION IMPROVEMENTS ---
    
    # A. Resize to original image size
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    
    # B. Thresholding: Remove weak activations to clean up noise
    # Any activation less than 15% of the max is zeroed out
    heatmap[heatmap < 0.15] = 0
    
    # C. Smoothing: Make the heatmap look organic
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    # D. Color Mapping with Masking
    # 1. Create the colored heatmap (JET)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 2. Convert original image to BGR for OpenCV
    original_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    
    # 3. SMART BLENDING (Fixes solid blue background)
    # Use the raw heatmap intensity (0-1) as a mask.
    # Where heatmap is 0, we add 0 color. Where heatmap is 1, we add full color.
    mask = heatmap[:, :, np.newaxis] # Shape (H, W, 1)
    
    # Formula: Original + (HeatmapColor * IntensityMask * Strength)
    superimposed_img = original_cv.astype(np.float32) + (heatmap_colored.astype(np.float32) * mask * 0.6)
    
    # Clip to valid range [0, 255] and convert back to uint8
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    output_filename = "gradcam_result.png"
    cv2.imwrite(output_filename, superimposed_img)
    print(f"Grad-CAM saved to {output_filename}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run chest_trainer.py first.")
    else:
        model = load_model_for_inference(MODEL_PATH)
        
        # Pick a random image
        all_images = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
        if all_images:
            random_img = random.choice(all_images)
            visualize(os.path.join(IMG_DIR, random_img), model)
        else:
            print("No images found in images/ folder.")     