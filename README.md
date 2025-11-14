## Real-Time Face Mask Detector (EfficientNetB0)
---
## Overview

This project implements a binary image classification model to detect whether a person is wearing a face mask or not, using transfer learning with EfficientNetB0.
The goal is to achieve high accuracy with efficient inference for real-time deployment. 

---
## Project Summary

| Field       | Methodology                      |
|-------------|-----------------------------------|
| **Domain**  | Computer Vision / Deep Learning   |
| **Task**    | Binary Image Classification       |
| **Model**   | EfficientNetB0 (pretrained on ImageNet) |
| **Framework** | TensorFlow / Keras             |
| **Dataset** | Custom dataset with `with_mask` and `without_mask` folders |
| **Evaluation** | Accuracy, Validation Accuracy, Training Curves |

---
## Model Architecture
### Base Model
- EfficientNetB0 — pretrained on ImageNet
- Used as a feature extractor
- Layers frozen during initial training, then partially unfrozen for fine-tuning

### Custom Top Layers
 ``` python
Input → EfficientNetB0 → GlobalAveragePooling2D → Dropout(0.3) → Dense(2, softmax)
```
### Training Strategy
- **Phase 1**: Train only top layers (frozen base)
- **Phase 2**: Fine-tune top 200 layers of EfficientNetB0
- **Callbacks** : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---
### Training Setup
 ``` python
- IMG_SIZE = 224
- BATCH_SIZE = 32
- EPOCHS = 20
```
### Data Augmentation:
- Rotation, Width/Height shift
- Shear, Zoom, Flip
- Rescaling (1./255)

**Optimizer** : Adam (1e-4 → 1e-5 during fine-tuning)

**Loss** : Categorical Crossentropy

**Metrics** : Accuracy

---
##  Results

| Metric               | Value (approx.)        |
|----------------------|-------------------------|
| **Train Accuracy**   | ~98%                    |
| **Validation Accuracy** | ~97%                |
| **Fine-Tuned Accuracy** | >98%                |
| **Loss**             | Stable, non-divergent   |

## Observation:
EfficientNetB0 achieves strong accuracy with fast convergence and low overfitting after augmentation and learning rate scheduling.

## Visualization
<img width="595" height="488" alt="image" src="https://github.com/user-attachments/assets/7a5f401d-ff9f-4489-99e7-f3c702243393" />
<img width="595" height="492" alt="image" src="https://github.com/user-attachments/assets/5682923c-e2a9-49f9-9800-52b1ebd3c381" />

## Single Image Prediction
<img width="369" height="505" alt="image" src="https://github.com/user-attachments/assets/3512ca12-0e9f-44d2-8a6d-659441f25e02" />

---

## 🧑‍💻 Author
Mourad Sleem

moradbshina@gmail.com

