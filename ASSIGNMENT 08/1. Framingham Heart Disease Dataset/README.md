# Framingham Heart Disease Dataset

Classification of whether a person is likely to develop Coronary Heart Disease (CHD) within the next 10 years.

## Files

- `framingham.csv` - the dataset
- `Framingham_Heart_Disease_Classification.ipynb` - source code / solution notebook

## About the Dataset

- Source: Framingham Heart Study (Kaggle)
- **4240 rows** and **15 feature columns** + target
- Target column: `TenYearCHD`
  - `0` = not likely to develop CHD
  - `1` = likely to develop CHD within 10 years
- Features are medical / lifestyle indicators: age, blood pressure, cholesterol, smoking status, BMI, glucose, etc.

## Pipeline Used

1. Loaded the data and performed basic analysis (`df.info()`, `describe()`).
2. **Missing values:** `education` filled with mode, numeric columns filled with median.
3. **Scaling:** `StandardScaler` (fit on train only, to prevent data leakage).
4. **Balancing:** `SMOTE` applied on training data (dataset is ~85% / ~15% imbalanced).
5. **Models:** trained and compared 7 classifiers (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes) on Accuracy, Precision, Recall and F1-Score.
6. **Best model:** Random Forest.
7. **Hyperparameter tuning:** `GridSearchCV` on Random Forest.

## Results

| Metric | Default Random Forest | Tuned Random Forest |
|--------|----------------------|---------------------|
| Accuracy | 0.8054 | 0.8007 (CV: 0.9092) |

The tuned model's test accuracy stayed around 0.80 (common for highly imbalanced medical data), but the cross-validation score improved to ~0.91, meaning the tuned model generalizes better.

Most important features: `age`, `sysBP`, `cigsPerDay`, `glucose`.

## How to Run

1. Install requirements: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter`
2. Open the notebook in Jupyter and run all cells (`Kernel -> Restart & Run All`).
