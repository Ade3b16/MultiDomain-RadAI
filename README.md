# MultiDomain-RadAI: AI-Powered Multi-Domain Radiological Analysis and Automated Reporting

## 💡 Project Overview

**MultiDomain-RadAI** is a Master's level AI engineering project designed to build a robust, end-to-end system for automated analysis and diagnostic reporting of medical X-ray images across multiple anatomical domains.

The system addresses radiologist workload and diagnostic variability using a cascaded deep-learning pipeline that mimics human expert reasoning:

**Identify the image → Analyze the pathology → Generate a structured report**

---

## 🎯 Key Goals and Features

### 1. Multi-Domain Generalization (Phase 1)
Train a single **Gatekeeper** model to classify X-rays by body part (Chest, Knee, Hand, Skull, etc.) and route them to the correct specialized module.

### 2. Pathology Localization and Classification (Phase 2)
Use **Multi-Task Learning (MTL)** to classify abnormalities (e.g., Pneumonia, Fracture) and generate localization heatmaps.

### 3. Evidence-Grounded Reporting (Phase 3)
Apply a **Vision-Language Model (VLM)** to translate model outputs into structured, clinically meaningful diagnostic reports.

### 4. Explainable AI (XAI)
Integrate **Grad-CAM** visualizations to show which regions influenced the model’s decisions.

---

## ⚙️ Proposed Architecture: Three-Phase Cascade

### Phase 1: Image Gatekeeping (Classification & Routing)
- **Purpose:** Identify the anatomical region.  
- **Model:** Pre-trained CNN (ResNet/EfficientNet) fine-tuned on UNIFESP.  
- **Output:** Body part label (e.g., `Chest`, `Knee`, `Hand`).  

### Phase 2: Deep Analysis (Multi-Task Vision)
- **Purpose:** Detect, classify, and localize abnormalities.  
- **Model:** DenseNet-121 optimized for Thoracic and Musculoskeletal categories.  
- **Tasks:**  
  - Multi-label disease classification  
  - Weakly-supervised localization via Grad-CAM  

### Phase 3: Automated Reporting (Vision-Language Integration)
- **Purpose:** Generate structured diagnostic narratives.  
- **Model:** Transformer Decoder / VLM conditioned on Phase-2 outputs.  
- **Output:** `Findings` and `Impression` sections.  

---

## 💾 Datasets Used

- **UNIFESP X-Ray Body Part Classification** – Phase 1  
- **CheXpert / NIH ChestXray14** – Thoracic analysis & reporting  
- **MURA** – Musculoskeletal analysis  

---

## 📈 Evaluation Metrics

| Module | Metrics |
|--------|---------|
| **Classification** | AUC |
| **Localization** | IoU (Jaccard), DICE |
| **Reporting** | Clinical F1-Score, BLEU, ROUGE |

---

## 💻 Repository Structure & Reproducibility

This repository will include:

- Full implementation  
- Training and inference scripts  
- Environment specifications  
- Experiment documentation  
- Milestone submissions (M2, M3, Final Report)

---

## 🔗 Tentative Repository Link

[https://github.com/Ade3b16/MultiDomain-RadAI]
