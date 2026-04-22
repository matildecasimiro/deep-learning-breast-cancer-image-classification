# 🧠 Breast Cancer Detection using Deep Learning

This repository contains the implementation and results of a deep learning project developed for the **Deep Learning course (2024/2025)** in the Bachelor’s Degree in Data Science at NOVA IMS.

The goal of this project is to support breast cancer diagnosis by automatically classifying histopathological images using deep learning techniques.

---

## 📌 Project Overview

Breast cancer is one of the leading causes of death among women worldwide. Early and accurate diagnosis is crucial for improving patient outcomes.

In this project, we use the **BreaKHis dataset**, which contains high-resolution microscopic images of breast tissue, to build models capable of:

- **Binary Classification**: Distinguish between *benign* and *malignant* tumors  
- **Multi-Class Classification**: Identify the specific tumor type among 8 classes  

---

## 📂 Dataset

- **Name**: BreaKHis (Breast Cancer Histopathological Database)  
- **Content**:
  - Microscopic images of breast tissue  
  - Labels:
    - Benign / Malignant  
    - Tumor types:
      - Adenosis  
      - Fibroadenoma  
      - Phyllodes Tumor  
      - Tubular Adenoma  
      - Ductal Carcinoma  
      - Lobular Carcinoma  
      - Mucinous Carcinoma  
      - Papillary Carcinoma  
  - Magnification levels  

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handling missing values
- Label encoding for binary and multi-class tasks
- Stratified train-test split
- Image preprocessing:
  - Resizing (50x50)
  - Normalization
  - Duplicate removal

### 2. Image Transformations
- Grayscale conversion  
- RGB scaling  
- Contrast adjustment  
- Laplacian filtering  

### 3. Data Augmentation
- Performed using `ImageDataGenerator` to reduce overfitting and improve generalization

---

## 🧠 Models

### Binary Classification
- CNN built from scratch  
- Transfer learning:
  - VGG16  
  - ResNet50  

- Techniques used:
  - Adam optimizer  
  - Binary cross-entropy loss  
  - AUC-PR metric  
  - Early stopping (callbacks)  
  - Hyperparameter tuning with HyperBand  

### Multi-Class Classification
- CNN (from scratch)  
- VGG16 (transfer learning)  
- Functional API model (image + binary label input)  

- Techniques used:
  - Sparse categorical cross-entropy  
  - Class weighting (to address imbalance)  
  - Hyperparameter tuning  
  - Accuracy, F1-score, Precision, Recall evaluation  

---

## 📊 Results

### Binary Classification (Best Model)
- **Model**: CNN with RGB scaling  
- **F1 Score**: 85%  
- **Recall**: 86%  
- **Precision**: 85%  

### Multi-Class Classification (Best Model)
- **Model**: Functional API with contrast adjustment  
- **F1 Score**: 58%  
- **Recall**: 63%  
- **Precision**: 59%  

---

## 📉 Error Analysis

- Binary model:
  - Strong performance on malignant detection  
  - Slight bias toward predicting malignant tumors  

- Multi-class model:
  - Struggles with minority classes (e.g., Phyllodes Tumor)  
  - Confusion between similar tumor types  

---

## 🚀 Future Work

- Apply ensemble methods (e.g., stacking models)  
- Use more advanced architectures  
- Test on external datasets for better generalization  
- Improve handling of class imbalance  

---

## 🛠️ Technologies Used

- Python 3  
- TensorFlow / Keras  
- Keras Tuner  
- OpenCV (`cv2`)  
- NumPy  
- Pandas  
- Matplotlib / Seaborn  

---

## 📁 Repository Structure (example)

├── project_deep_learning.ipynb # Jupyter notebooks for model development
├── utils.py # functions used
├── Deep_Learning_Report.pdf
├── README.md

---

## 👥 Authors

- Laura Matias  
- Marta Aliende  
- Marta Almendra  
- Matilde Casimiro  
- Teresa Simão  

---

## ⚠️ Disclaimer

This project is for academic purposes only and is not intended for clinical use.

