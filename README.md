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

p
