# Breast Cancer Wisconsin Classifier

## a. Problem statement
This project aims to build and compare different supervised machine learning models for classifying breast cancer as malignant or benign based on tumor features. The objective is to evaluate which classifier performs best in terms of prediction accuracy and classification quality using medical diagnostic data.

The problem is a binary classification task in which the model predicts whether a tumor is malignant (M) or benign (B) from numerical features extracted from digitized breast mass images.

## b. Dataset description
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository.

- Number of samples: 569
- Number of features: 30 numeric features
- Target variable: Diagnosis
- Class labels:
  - M = Malignant
  - B = Benign
- Feature types: Real-valued measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension

This dataset is commonly used for evaluating supervised learning algorithms in medical diagnosis problems because it contains enough information to distinguish malignant and benign tumors based on measurable characteristics.

## c. GitHub Repository Link
GitHub Repository: https://github.com/gadesowjanya/ml_assignment_2

## d. Models used
The following five machine learning models were implemented and compared:

- Logistic Regression
- Decision Tree
- KNN
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

## Model Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

From the comparison table, the Support Vector Machine (SVM) achieved the highest accuracy and F1 score, making it the best-performing model among the five classifiers tested in this assignment.

## Project setup and execution
1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the environment:

   On macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   On Windows Command Prompt:

   ```cmd
   .venv\Scripts\activate.bat
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## App usage
This application compares the performance of multiple classifiers and allows the user to select a trained model for prediction on a custom CSV dataset. A template CSV file can be downloaded from the app to ensure the required feature columns are included.

The system is designed for educational use only and should not be used for medical diagnosis or treatment decisions.