# California Housing Dataset

The California Housing dataset (normally used for regression) converted into a **binary classification** problem: predict whether a house is **expensive** (price above the dataset median) or **affordable** (price below the median).

## Files

- `housing.csv` - the dataset
- `Housing_Price_Classification.ipynb` - source code / solution notebook

## About the Dataset

- **20640 rows** and **10 columns**
- `ocean_proximity` is a categorical column (one-hot encoded)
- `median_house_value` is used to create the target `price_category`
  - `1` = Expensive (price >= dataset median)
  - `0` = Affordable (price < dataset median)
- `total_bedrooms` has 207 missing values (filled with median)
- Classes are **almost perfectly balanced** (50 / 50), so no SMOTE needed.

## Pipeline Used

1. Loaded the data and performed basic analysis (`df.info()`, `describe()`).
2. **Missing values:** `total_bedrooms` filled with median.
3. **Target creation:** `price_category` created from `median_house_value`.
4. **Encoding:** one-hot encoding of `ocean_proximity` with `drop_first=True` to avoid the dummy variable trap.
5. **Scaling:** `StandardScaler`.
6. **Models:** trained and compared 7 classifiers (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, Linear SVM, Naive Bayes) on Accuracy, Precision, Recall and F1-Score.
7. **Best model:** Random Forest.
8. **Hyperparameter tuning:** `GridSearchCV` on Random Forest (3-fold CV due to dataset size).

## Results

| Metric | Default Random Forest | Tuned Random Forest |
|--------|----------------------|---------------------|
| Accuracy | 0.8939 | 0.8944 |

The improvement after tuning is small because Random Forest was already performing very well (~0.89). `median_income` is the most important feature, matching real-world expectations.

## How to Run

1. Install requirements: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`
2. Open the notebook in Jupyter and run all cells (`Kernel -> Restart & Run All`).
