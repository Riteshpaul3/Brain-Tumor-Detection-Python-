# 🧠 Brain Tumor Detection Using CNN with Explainable AI and ViT Comparison

## 📌 Overview

This project focuses on detecting **brain tumors from MRI scans** using a **Convolutional Neural Network (CNN)**. To make the model predictions more transparent and interpretable, we integrate **Explainable AI (XAI)** tools such as:

- 🔍 SHAP (SHapley Additive Explanations)
- 🧩 LIME (Local Interpretable Model-Agnostic Explanations)
- 🔥 Grad-CAM (Gradient-weighted Class Activation Mapping)

To benchmark performance, we also compare the CNN model against a **Vision Transformer (ViT)** architecture. All experiments are run on **Google Colab**, utilizing GPU for fast training.

---

## ✅ Key Features

- 🎯 High-accuracy brain tumor classification using CNN
- 🧠 Interpretability via SHAP, LIME, and Grad-CAM
- 📊 Model comparison with Vision Transformer
- 📸 Visual outputs for prediction and explanation
- ☁️ 100% compatible with Google Colab

---

---

## 🧠 Models Used

### 1. CNN (Convolutional Neural Network)
- Built using Keras
- Includes Conv2D, MaxPooling, Flatten, Dense, Dropout layers
- Classifies MRI images as `Tumor` or `No Tumor`

### 2. Vision Transformer (ViT)
- Transformer-based image classification model
- Operates on image patches instead of convolutions
- Used for performance comparison with CNN

### 3. Explainable AI Tools

| Method     | Purpose                                  | Output                  |
|------------|------------------------------------------|--------------------------|
| SHAP       | Global + local feature importance         | Value-based overlays     |
| LIME       | Local explanation using superpixels       | Segmented masks          |
| Grad-CAM   | Highlights most important image regions   | CNN-based heatmaps       |

---

## 🧪 Results

### 🔢 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| CNN (Baseline)      | 98.5%    | 98.7%     | 98.4%  | 98.5%    |
| CNN + SHAP          | 98.5%    | 98.7%     | 98.4%  | 98.5%    |
| CNN + LIME          | 98.5%    | 98.6%     | 98.3%  | 98.4%    |
| CNN + Grad-CAM      | 98.5%    | 98.8%     | 98.4%  | 98.6%    |
| Vision Transformer  | 97.2%    | 97.4%     | 97.0%  | 97.2%    |

### 📊 Explainability Summary

| Tool      | Pros                                               | Cons                                        |
|-----------|----------------------------------------------------|---------------------------------------------|
| SHAP      | Pixel-level importance, model-agnostic             | Slower with large images                    |
| LIME      | Simple visual output, works with any model         | Inconsistent with different segmentations   |
| Grad-CAM  | Effective for CNN heatmaps                         | Not compatible with ViT                     |

✅ **Conclusion**: The CNN + Grad-CAM combination provided the most interpretable and accurate results.

---

## 📂 Dataset

- **Name**: Brain MRI Images for Brain Tumor Detection  
- **Source**: [Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)  
- **Classes**: Tumor, No Tumor  
- **Preprocessing**:
  - Image resizing to uniform shape
  - Normalization
  - One-hot encoding of labels

---

## 🔧 How to Run (in Google Colab)

1. Open [Google Colab](https://colab.research.google.com/)
2. Mount Google Drive or upload dataset directly
3. Run the following notebooks in order:
   - `BrainTumorDetection_CNN.ipynb`
   - `shap_explanation.ipynb`
   - `lime_explanation.ipynb`
   - `gradcam_explanation.ipynb`
   - `ViT_Comparison.ipynb`
4. View performance metrics and explanation visualizations in the output cells.

---
## 📚 References

- [🔗 SHAP](https://github.com/slundberg/shap)
- [🔗 LIME](https://github.com/marcotcr/lime)  
- [🔗 Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)  
- [🔗 Vision Transformer (ViT)](https://github.com/lucidrains/vit-pytorch)
- [🔗 Brain MRI Dataset (Kaggle)](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)


## 📦 Requirements

Most packages are pre-installed in Google Colab. For local use:

```bash
pip install tensorflow keras numpy matplotlib opencv-python shap lime
