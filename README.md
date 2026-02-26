# 🏦 Loan Status Prediction using Deep Learning and K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## 📌 Overview
This repository contains a hybrid machine learning pipeline designed to predict loan approval statuses. The project innovatively combines unsupervised learning (K-Means Clustering) for feature engineering with a supervised Deep Learning model (Multi-Layer Perceptron) built in PyTorch. 

To ensure robust model performance, the pipeline includes comprehensive data preprocessing, handling of class imbalances using SMOTE, and extensive exploratory data analysis (EDA).

## 🚀 Key Features
* **Automated Preprocessing:** Handles missing values and label-encodes categorical variables automatically.
* **Feature Engineering via Clustering:** Uses K-Means clustering to discover hidden patterns in the applicant data, adding the resulting cluster labels as a new feature to improve predictive power.
* **Class Imbalance Handling:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to oversample the minority class, preventing model bias.
* **Deep Learning Classifier:** A robust 4-layer MLP built with PyTorch, utilizing Batch Normalization and Dropout for regularization.

## 📊 Exploratory Data Analysis & Clustering

### 1. Determining Optimal Clusters (Elbow Method & Silhouette Scores)
Before feeding data into the neural network, we analyze the dataset using K-Means. The Elbow Method and Silhouette plots help us determine the optimal number of clusters.


<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/d5fd1823-f6a4-4b3a-8cb2-6d892332fdb8" />


<img width="551" height="414" alt="image" src="https://github.com/user-attachments/assets/64714c47-6b46-45cd-a0ab-022db5e1183b" /> <img width="551" height="414" alt="image" src="https://github.com/user-attachments/assets/ca9b41ce-1bf3-4bb7-a9b6-35743d2ec6fe" />



### 2. Correlation Analysis
Understanding how features interact with each other and the target variable.

<img width="886" height="803" alt="image" src="https://github.com/user-attachments/assets/4a61fb0b-4b6a-4559-87d6-5ae8fa21d83d" />


### 3. Target Class Distribution
Visualizing the dataset before applying SMOTE to understand the baseline class imbalance.

<img width="695" height="374" alt="image" src="https://github.com/user-attachments/assets/215d44fe-f9be-4858-a67b-500ebef7db16" />


## 🧠 Model Architecture & Performance

The predictive model is a Multi-Layer Perceptron (MLP) defined as follows:
* **Input Layer:** Dynamic based on features + cluster label
* **Hidden Layers:** 64 -> 32 -> 16 neurons with ReLU activation, Batch Normalization, and 50% Dropout.
* **Output Layer:** 1 neuron with Sigmoid activation for binary classification.

### Results
After training for 50 epochs, the model is evaluated on a stratified test set. 

> 📸 **Add your Confusion Matrix Heatmap here:**
![Confusion Matrix](docs/images/confusion_matrix.png)

*(Note: You can manually type your final Accuracy, F1 Score, Precision, and Recall scores here once you run the final script!)*
* **Accuracy:** `0.XXXX`
* **F1 Score:** `0.XXXX`
* **Precision:** `0.XXXX`
* **Recall:** `0.XXXX`

## 🛠️ Tech Stack
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `imbalanced-learn` (SMOTE)
* **Deep Learning:** `PyTorch`
* **Data Visualization:** `matplotlib`, `seaborn`

## ⚙️ How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/loan-status-prediction.git](https://github.com/your-username/loan-status-prediction.git)
