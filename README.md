# Breast Cancer Classification — ML Assignment 2

## a. Problem Statement
Breast cancer diagnosis is traditionally made by examining cell nuclei
characteristics from a digitized image of a fine needle aspirate (FNA) of a
breast mass. The goal of this project is to build and compare multiple
machine learning classifiers that predict whether a tumor is **malignant**
or **benign** from 30 numeric features describing cell nuclei, and to expose
the trained models through an interactive Streamlit web application.

## b. Dataset Description
- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** Originally donated to the UCI Machine Learning Repository by
  Dr. William H. Wolberg, University of Wisconsin Hospitals; also available
  via `sklearn.datasets.load_breast_cancer`.
- **Instances:** 569 (≥ 500 required)
- **Features:** 30 numeric features (≥ 12 required) — mean, standard
  error, and "worst" values of 10 real-valued measurements per cell nucleus
  (radius, texture, perimeter, area, smoothness, compactness, concavity,
  concave points, symmetry, fractal dimension).
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign
- **Split used:** 80% train / 20% test, stratified, `random_state=42`

## c. GitHub Repository Link
`https://github.com/jenishjoyal-99/multi-classification-models.git`

## d. Models Used

All 5 models were trained on the same 80/20 stratified train/test split of
the dataset above, with features standardized via `StandardScaler`.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9932 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset. After standardization the classes are close to linearly separable, so a simple linear decision boundary generalizes very well — highest accuracy, AUC, F1, and MCC of all five models, with only 2 misclassifications out of 114 test samples. |
| Decision Tree | Weakest performer. A single unconstrained tree overfits the training split (deep splits chase noise in individual features), which hurts recall on the malignant class in particular. Would likely improve with depth/leaf-size pruning or cost-complexity pruning. |
| kNN | Strong performance, tied with Random Forest on Accuracy/F1. Because features are scaled before distance computation, the 5-nearest-neighbor vote reflects genuine local structure in the data. Slightly lower AUC than the ensemble/linear models since raw class probabilities from KNN are coarse (only 6 possible probability values with k=5). |
| Naive Bayes | Solid, fast baseline despite the (unrealistic) assumption that all 30 features are conditionally independent given the class. Notably it has the second-highest AUC, meaning its probability ranking is very good even though its hard-label accuracy is a bit lower than Logistic Regression's. |
| Random Forest (Ensemble) | Second-best overall, and the most robust: averaging 200 trees fixes the overfitting problem seen in the single Decision Tree and pushes AUC to 0.9932, the highest among all models. Marginally behind Logistic Regression only because this dataset is close to linearly separable, which favors a linear model. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest Accuracy (0.9825), AUC (0.9954), F1 (0.9861) and MCC (0.9623). Random Forest is a close second and would likely be the safer choice on noisier or less linearly-separable data. |

## Project Structure
```
project-folder/
│-- app.py                 # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv          # held-out test split used for evaluation/demo
│-- model/
│   │-- train_models.py    # trains all 5 models + saves artifacts
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- scaler.pkl
│   │-- feature_names.json
│   │-- metrics_comparison.csv
```

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models + test_data.csv (optional, already included)
streamlit run app.py
```

## Live App
`https://multi-classification-models-qkerqx2trseucwf3qyatre.streamlit.app/`
