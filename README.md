# CellScope: Breast Tumour Classification Studio

## a. Problem statement
This project compares five supervised classification algorithms for predicting the diagnostic class associated with breast mass measurements. The target is binary: 0 = malignant and 1 = benign. This is an educational machine-learning exercise, not a medical diagnostic product.

## b. Dataset description
- Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), originally made publicly available through the UCI Machine Learning Repository and distributed by scikit-learn's dataset utilities.
- Instances: 569.
- Predictors: 30 numeric features computed from digitized fine needle aspirate images.
- Target: binary classification (0 = malignant, 1 = benign).
- Split: stratified 80% training and 20% test; random_state=27.
- The supplied `dataset.csv` contains all 569 rows. `test_data.csv` contains the held-out test subset.
- Preprocessing is performed within model pipelines to avoid fitting transformations on the test data.

## c. GitHub Repository Link
[ADD YOUR PUBLIC GITHUB REPOSITORY URL HERE]

## d. Models used and evaluation
All five classifiers use the same training and test partitions. Logistic Regression, KNN and SVM use standardization. Median imputation is included in pipelines for reproducibility.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9980 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9474 | 0.9630 | 0.9714 | 0.9444 | 0.9577 | 0.8886 |
| K-Nearest Neighbors | 0.9737 | 0.9858 | 0.9726 | 0.9861 | 0.9793 | 0.9433 |
| Gaussian Naive Bayes | 0.9386 | 0.9868 | 0.9114 | 1.0000 | 0.9536 | 0.8715 |
| Support Vector Machine | 0.9912 | 0.9980 | 0.9863 | 1.0000 | 0.9931 | 0.9812 |


### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.9825, AUC 0.9980, precision 0.9861, recall 0.9861, F1 0.9861, MCC 0.9623. Interpret relative to the other models and the class-specific error costs. |
| Decision Tree | Accuracy 0.9474, AUC 0.9630, precision 0.9714, recall 0.9444, F1 0.9577, MCC 0.8886. Interpret relative to the other models and the class-specific error costs. |
| K-Nearest Neighbors | Accuracy 0.9737, AUC 0.9858, precision 0.9726, recall 0.9861, F1 0.9793, MCC 0.9433. Interpret relative to the other models and the class-specific error costs. |
| Gaussian Naive Bayes | Accuracy 0.9386, AUC 0.9868, precision 0.9114, recall 1.0000, F1 0.9536, MCC 0.8715. Interpret relative to the other models and the class-specific error costs. |
| Support Vector Machine | Accuracy 0.9912, AUC 0.9980, precision 0.9863, recall 1.0000, F1 0.9931, MCC 0.9812. Interpret relative to the other models and the class-specific error costs. |
| Overall winner | **Support Vector Machine** has the highest AUC on this fixed holdout split (F1 used as a tie-breaker). This is a split-specific result, not proof of universal superiority. |

Add your own interpretation of the confusion matrices and ROC curves after executing the notebook in BITS Virtual Lab. In particular, discuss false negatives for the malignant class and why recall/sensitivity matters in a medical screening context.

## Streamlit application
- Main application: `app.py`
- Features: labeled CSV upload, model-selection dropdown, six evaluation metrics, confusion matrix, classification report, and side-by-side comparison of all five models.
- Live app URL: [ADD YOUR STREAMLIT COMMUNITY CLOUD URL HERE]

## Reproducibility
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook `ML_Assignment_2.ipynb` from top to bottom to regenerate model artifacts and `test_data.csv`.
3. Start the application: `streamlit run app.py`

## Limitations
The dataset is relatively small and historical. A random holdout evaluation is not external clinical validation. Predictions must not be used for diagnosis or treatment decisions.
