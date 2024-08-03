# Diabetes Prediction and Analysis
This project is part of a machine learning course at university and involves analyzing a diabetes dataset to build a predictive model for diabetes based on various health metrics.

## Data Preprocessing and Visualization
### Reading Dataset and Summary Statistics
- The dataset is read using pandas, and summary statistics are printed to understand the data distribution.
- Missing values in features (except for the target variable Diabetic) are imputed with the median value of the respective feature.
### Distribution of `Diabetic` Class
- A Kernel Density Estimate (KDE) plot is generated to visualize the distribution of the Diabetic class.
### Histograms for Number of Diabetics in Each Age Group
- A histogram is plotted to show the number of diabetics in different age groups.
### Density Plots for AGE and BMI
- KDE plots are generated for the `AGE` and `BMI` features to visualize their distributions.
### Visualizing Correlations
- A heatmap is created to visualize the correlations between input features and the Diabetic class.
## Linear Regression Models
A linear regression models are built to predict the AGE feature based on various input features.
### Models Details
- **Model LR1**: Uses all features except `AGE` to predict `AGE`.
- **Model LR2**: Uses the NPG feature to predict `AGE`.
- **Model LR3**: Uses the `NPG`, `DIA`, and `PGL` features to predict `AGE`.
### Evaluation
- The Mean Squared Error (MSE) and R-squared values are calculated for each model.
- Scatter plots are generated to visualize actual vs. predicted `AGE` values.
## k-Nearest Neighbors Classification
The k-Nearest Neighbors (kNN) classifier is used to predict whether a person is diabetic.
### Model Details
- Different values of `k` and distance metrics (`euclidean`, `manhattan`, `chebyshev`) are evaluated.
- Performance metrics such as precision, recall, F1 score, accuracy, ROC/AUC score, and confusion matrix are calculated and visualized.
### Evaluation
- ROC curves and confusion matrices are plotted for different values of `k` and metrics to analyze the model's performance.
