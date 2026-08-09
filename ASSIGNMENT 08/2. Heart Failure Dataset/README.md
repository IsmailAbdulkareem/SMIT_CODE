# Heart Failure Dataset

Classification of whether a heart failure patient will **die** or **survive** based on their clinical records.

## Files

- `heart_failure_clinical_records_dataset.csv` - the dataset
- `Heart_Failure_Classification.ipynb` - source code / solution notebook

## About the Dataset

- Source: Kaggle
- **299 patient records** and **13 feature columns** + target
- Target column: `DEATH_EVENT`
  - `1` = patient deceased
  - `0` = patient survived
- The dataset has **no missing values**.
- Class distribution: ~32% deceased, ~68% survived (imbalanced, handled with SMOTE).

## Pipeline Used

1. Loaded the data and performed basic analysis (`df.info()`, `describe()`).
2. **EDA:** target distribution, boxplots, group-by means, correlation matrix. Patients who died were on average older, with lower ejection fraction and serum sodium, and higher serum creatinine.
3. **Scaling:** `StandardScaler`.
4. **Balancing:** `SMOTE` applied on training data only.
5. **Models:** trained and compared 7 classifiers (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes) on Accuracy, Precision, Recall and F1-Score.
6. **Best model:** Random Forest (after tuning).
7. **Hyperparameter tuning:** `GridSearchCV` on Random Forest.

## Results

| Metric | Logistic Regression (initial best) | Tuned Random Forest |
|--------|-----------------------------------|---------------------|
| Accuracy | 0.8167 | 0.8500 |

The tuned Random Forest reached **0.85 accuracy** with better Precision, Recall and F1-Score than all other models.

Most important features: `serum_creatinine`, `age`, `ejection_fraction`, `serum_sodium`.

## How to Run

1. Install requirements: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter`
2. Open the notebook in Jupyter and run all cells (`Kernel -> Restart & Run All`).
