# Breast Cancer Wisconsin Classifier

## a. Problem statement
This project focuses on developing a supervised machine learning system to classify breast tumors as malignant or benign using diagnostic measurements extracted from breast cancer imaging data. The objective is to evaluate the effectiveness of several classifiers and identify the model that delivers the best predictive performance for this medical classification problem.

The task is a binary classification problem, where the model predicts whether a tumor is malignant (M) or benign (B) based on numerical features derived from the tumor characteristics.

## b. Dataset description
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository.

- Total samples: 569
- Number of features: 30
- Target variable: Diagnosis
- Classes:
  - M = Malignant
  - B = Benign
- Feature types: Real-valued measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension

This dataset is widely used in machine learning research because it contains meaningful feature patterns that help distinguish malignant and benign tumors with high accuracy.

## c. GitHub Repository Link
GitHub Repository: https://github.com/gadesowjanya/ml_assignment_2

## d. Models used
The following five machine learning models were implemented and evaluated:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
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

Based on the comparison, the Support Vector Machine (SVM) achieved the highest Accuracy and F1-score, indicating the strongest overall performance among the five models evaluated in this assignment.

## Model performance observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved an Accuracy of 0.900 and a perfect Recall of 1.000, indicating that it correctly identified all malignant cases. However, its Precision of 0.833 shows that one benign case was misclassified as malignant, resulting in a false positive. |
| Decision Tree | The Decision Tree achieved perfect results across all evaluation metrics, including Accuracy, Precision, Recall, and F1-score, with values of 1.000. It correctly classified every malignant and benign instance in the test dataset without error. |
| KNN | KNN produced results similar to Logistic Regression, with an Accuracy of 0.900, Precision of 0.833, and Recall of 1.000. It successfully detected all actual malignant cases but misclassified one benign sample due to the local neighborhood structure in the small test set. |
| Naive Bayes | Naive Bayes achieved an Accuracy of 0.900, Precision of 0.833, and Recall of 1.000. It identified all malignant cases correctly, but it also produced the same single false positive observed in the other competing models. |
| Support Vector Machine | The Support Vector Machine achieved an Accuracy of 0.900 and Recall of 1.000, showing strong detection of malignant tumors. Although it correctly separated the malignant region, it still allowed one benign case to fall within the decision boundary, resulting in a false positive. |

### Overall winner for this dataset
**Decision Tree** is the best-performing model for this specific dataset, achieving a perfect **F1-score of 1.000** because the dataset was small. Although Decision Tree is more prone to overfitting than SVM or Logistic Regression for bigger dataset.

## Project execution
1. Open the Streamlit application using the following link:  
   https://breast-cancer-mlassignment2.streamlit.app/

2. Upload the file named `sample_prediction.csv` from the repository.

3. Select the model to generate predictions and evaluate the results.
