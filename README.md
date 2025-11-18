MultiDomain-RadAI: AI-Powered Multi-Domain Radiological Analysis and Automated Reporting

💡 Project Overview

MultiDomain-RadAI is a Master's level AI engineering project designed to create a robust, end-to-end system for the automated analysis and diagnostic reporting of medical X-ray images across multiple anatomical domains.

The system addresses the clinical challenge of radiologist workload and diagnostic variability by implementing a cascaded deep learning pipeline that mirrors the logical steps of a human expert: Identify the image $\rightarrow$ Analyze the pathology $\rightarrow$ Generate a structured report.

🎯 Key Goals and Features

Multi-Domain Generalization (Phase 1): Train a single "Gatekeeper" model capable of classifying X-rays across various body parts (Chest, Knee, Hand, Skull, etc.) and routing the image to the correct specialized analysis module.

Pathology Localization and Classification (Phase 2): Utilize Multi-Task Learning (MTL) to simultaneously classify diseases/abnormalities (e.g., Pneumonia, Fracture) and generate visual evidence (localization heatmaps) for enhanced interpretability.

Evidence-Grounded Reporting (Phase 3): Employ a Vision-Language Model (VLM) framework to translate the vision model's numerical and spatial outputs into a structured, clinically relevant diagnostic narrative, mitigating the risk of AI hallucination.

Explainable AI (XAI): Integrate Grad-CAM visualizations directly into the analysis output to show clinicians why a decision was made.

⚙️ Proposed Architecture: Three-Phase Cascade

Phase 1: Image Gatekeeping (Classification & Routing)

Model: Pre-trained CNN (ResNet/EfficientNet) fine-tuned on diverse X-ray datasets (UNIFESP).

Task: Multi-class classification of the body part (e.g., 'Chest', 'Knee', 'Elbow').

Phase 2: Deep Analysis (Multi-Task Vision)

Model: DenseNet-121 backbone optimized for specific domains (Thoracic, Musculoskeletal).

Task: Multi-label disease classification and weakly-supervised localization (Grad-CAM).

Phase 3: Automated Reporting (Vision-Language Integration)

Model: Transformer Decoder/VLM conditioned on Phase 2 outputs (labels, confidence, bounding boxes).

Task: Generation of structured 'Findings' and 'Impression' sections of a report.

💾 Datasets Used

This project uses several large, publicly available medical imaging datasets:

UNIFESP X-Ray Body Part Classification: Used for training the Phase 1 Gatekeeper model.

CheXpert / NIH Chest X-ray: Used for Phase 2 (Thoracic analysis) and Phase 3 (Report Generation).

MURA (Musculoskeletal Radiographs): Used for Phase 2 (Musculoskeletal analysis) and domain generalization testing.

📈 Evaluation Metrics

Performance is measured using clinically relevant metrics across the entire pipeline:

Classification: Area Under the ROC Curve (AUC)

Localization: Jaccard Index (IoU) / DICE Score

Reporting: Clinical Accuracy (F1-Score), BLEU/ROUGE Scores

💻 Repository Structure & Reproducibility

This repository will contain the full code, training scripts, environment specifications, and detailed documentation required to reproduce the results presented in the project milestones (M2, M3, Final Report).

Tentative Repository Link

[Your GitHub/GitLab Repository Link Here]