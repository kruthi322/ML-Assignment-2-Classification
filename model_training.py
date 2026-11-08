from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ARTIFACT_DIR = Path(__file__).resolve().parent

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, C=0.8, random_state=27))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(max_depth=4, min_samples_leaf=4, random_state=27))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GaussianNB(var_smoothing=1e-8))
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=1.5, kernel="rbf", probability=True, random_state=27))
        ])
    }

def train_and_export(data_path="dataset.csv", test_size=0.20, random_state=27):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["target"])
    y = data["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    out = []
    (ARTIFACT_DIR / "model").mkdir(exist_ok=True)
    for name, estimator in build_models().items():
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        score = estimator.predict_proba(X_test)[:, 1]
        out.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, score),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred)
        })
        artifact = {
            "Logistic Regression":"logistic_regression",
            "Decision Tree":"decision_tree",
            "K-Nearest Neighbors":"knn",
            "Gaussian Naive Bayes":"gaussian_nb",
            "Support Vector Machine":"svm"
        }[name]
        joblib.dump(estimator, ARTIFACT_DIR / "model" / f"{artifact}.joblib")
    result = pd.DataFrame(out)
    result.to_csv("model_metrics.csv", index=False)
    test = X_test.copy()
    test["target"] = y_test.values
    test.to_csv("test_data.csv", index=False)
    return result
