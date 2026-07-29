# Assignment 7: Linear Regression Implementations

## Part 1: Student Performance Prediction

Enhanced analysis predicting student exam scores using Linear Regression with improved feature engineering and visualization.

**Key Improvements Made:**
- Added polynomial features to capture non-linear relationships (study hours squared)
- Included interaction terms between study habits and demographic factors
- Implemented robust scaling for better model convergence
- Added prediction intervals and uncertainty quantification
- Enhanced visualizations with confidence bands and residual analysis plots
- Improved feature importance analysis using SHAP values

**Model Performance:**
- **R² Score**: 0.8989 (89.89% accuracy)
- **MAE**: 4.1302
- **MSE**: 25.9210
- **RMSE**: 5.0913

**Key Insights:**
- Study hours per day has the strongest positive correlation with exam scores
- Netflix/social media hours show negative impact on performance
- Interaction between study time and attendance is particularly significant
- Mental health ratings strongly influence exam outcomes

**Files:**
- `project.ipynb` - Enhanced implementation with comprehensive analysis
- `student_habits_performance.csv` - Student performance dataset

**Submission:** Complete repository on GitHub
