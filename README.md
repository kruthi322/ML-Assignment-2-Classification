# CellScope: Breast Tumour Classification Studio

## a. Problem Statement

This project compares five supervised machine learning classification algorithms for predicting the diagnostic class associated with breast mass measurements. The target variable is binary:

- `0` = Malignant
- `1` = Benign

The objective is to train, evaluate, and compare different classification models and identify the best-performing model based on multiple evaluation metrics.

This is an educational machine learning project and is not intended to be used as a medical diagnostic product.

## b. Dataset Description

- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
- **Source:** UCI Machine Learning Repository, distributed through scikit-learn dataset utilities.
- **Total instances:** 569.
- **Predictors:** 30 numerical features computed from digitized fine needle aspirate images.
- **Target:** Binary classification (0 = malignant, 1 = benign).
- **Data split:** Stratified 80% training and 20% testing.
- **Random state:** 27.

The `dataset.csv` file contains all 569 observations, while `test_data.csv` contains the held-out test subset.

Data preprocessing is performed using machine learning pipelines. Median imputation is included for handling missing values. Standardization is applied to Logistic Regression, K-Nearest Neighbors, and Support Vector Machine models.

## c. GitHub Repository Link

[ML Assignment 2 – Classification](https://github.com/kruthi322/ML-Assignment-2-Classification)

## d. Models Used and Evaluation

Five supervised machine learning classification algorithms were implemented and evaluated using the same training and testing partitions.

The following models were considered:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Evaluation Metrics

The models were evaluated using six performance metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Comparison of Model Performance

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9980 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9474 | 0.9630 | 0.9714 | 0.9444 | 0.9577 | 0.8886 |
| K-Nearest Neighbors | 0.9737 | 0.9858 | 0.9726 | 0.9861 | 0.9793 | 0.9433 |
| Gaussian Naive Bayes | 0.9386 | 0.9868 | 0.9114 | 1.0000 | 0.9536 | 0.8715 |
| Support Vector Machine | 0.9912 | 0.9980 | 0.9863 | 1.0000 | 0.9931 | 0.9812 |

### Observations on Model Performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 0.9825 and an AUC of 0.9980, indicating excellent classification performance and strong class discrimination. Its F1 score of 0.9861 and MCC of 0.9623 also demonstrate a strong balance between precision and recall.

**2. Decision Tree**

The Decision Tree classifier achieved an accuracy of 0.9474 and an AUC of 0.9630. It recorded the lowest AUC and MCC among the five models, indicating comparatively weaker overall performance on the test dataset.

**3. K-Nearest Neighbors**

KNN achieved an accuracy of 0.9737, an AUC of 0.9858, and an F1 score of 0.9793. It demonstrated strong classification performance and a good balance between precision and recall, although its overall performance was lower than SVM and Logistic Regression.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved an accuracy of 0.9386 and a recall of 1.0000. However, its precision of 0.9114 and MCC of 0.8715 were the lowest among the five models. Although it achieved perfect recall for the positive class under the default metric configuration, its overall classification performance was comparatively weaker.

**5. Support Vector Machine**

SVM achieved the highest accuracy of 0.9912, the highest F1 score of 0.9931, and the highest MCC of 0.9812. It also achieved an AUC of 0.9980, tied with Logistic Regression. These results indicate that SVM delivered the strongest overall performance on the selected test split.

### Overall Winner

**Support Vector Machine (SVM)** was identified as the overall best-performing model on this fixed test dataset.

It achieved the highest accuracy, F1 score, and MCC among all five classifiers. Logistic Regression also demonstrated excellent performance, with an AUC equal to that of SVM.

The selection is based on the evaluation results from this particular train-test split and does not establish universal superiority over other classification algorithms.

### Confusion Matrix and ROC Curve Analysis

The confusion matrices and ROC curves generated during notebook execution provide additional insight into the classification performance of the five models.

The confusion matrices help identify correctly and incorrectly classified observations, while ROC curves illustrate the trade-off between the true positive rate and false positive rate at different classification thresholds.

Since the target labels are 0 = malignant and 1 = benign, the confusion matrices should be examined carefully to identify false negatives for the malignant class. Missing a malignant case can have serious consequences in a medical screening context, making malignant-class sensitivity an important consideration.

The evaluation metrics should therefore be interpreted alongside the class-specific confusion matrix results rather than relying only on overall accuracy.

## Streamlit Application

The project includes an interactive Streamlit application for exploring and comparing the trained classification models.

- **Main application file:** `app.py`
- **Live application:** [CellScope – Breast Tumour Classification Studio](https://ml-assignment-2-classification-hmudkfua5fk4qtxu2awows.streamlit.app/)

### Application Features

- Upload a labeled CSV dataset.
- Select a classification model using the model-selection dropdown.
- View six evaluation metrics.
- Generate and view the confusion matrix.
- View the classification report.
- Compare the performance of all five classification models.

## Reproducibility

Follow the steps below to reproduce the project:

1. Clone or download the GitHub repository.
2. Install the required dependencies:

   `pip install -r requirements.txt`

3. Execute the Jupyter notebook `ML_Assignment_2(2).ipynb` from beginning to end to reproduce the model training and evaluation process.
4. Run the Streamlit application:

   `streamlit run app.py`

## Limitations

- The dataset contains 569 observations and is relatively small.
- The evaluation is based on a single stratified train-test split.
- The results do not represent external clinical validation.
- Model predictions should not be used independently for medical diagnosis or treatment decisions.

The project demonstrates the application of supervised machine learning techniques for binary classification, model evaluation, and interactive performance comparison.
