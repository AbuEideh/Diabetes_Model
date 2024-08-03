import pandas as pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, auc, roc_curve, \
    precision_score, recall_score, \
    f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Code for part one-1: Reading Dataset & Printing Summary Statistics
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(20)
dataSet = pandas.read_csv('Diabetes.csv')
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
dataset_imputed = dataSet.copy()
for feature in dataSet:
    if feature != 'Diabetic':
        median_value = dataSet[feature][dataSet[feature] != 0].median()
        dataset_imputed[feature] = dataset_imputed[feature].replace(0, median_value)
print(dataset_imputed.describe(include='all'))

#Code for part one-2: Distribution of `Diabetic` class
diabetic_column = 'Diabetic'
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.kdeplot(dataset_imputed[diabetic_column], fill=True)
plt.title(f'Kernel Density Estimate (KDE) for {diabetic_column}')
plt.xlabel(diabetic_column)
plt.ylabel('Density')
mean_value = dataset_imputed[diabetic_column].mean()
q3_value = dataset_imputed[diabetic_column].quantile(0.75)
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(q3_value, color='orange', linestyle='dashed', linewidth=2, label='3rd Quantile (75%)')
plt.legend()
plt.show()


#Code for part one-3: Histograms for number of diabetics in each age group
age_column = 'AGE'
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.histplot(dataset_imputed, x=age_column, hue=diabetic_column, multiple='stack', edgecolor='black', bins=50)
plt.title(f'Histogram of Diabetics in Each Age Group')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


#Code for part one-4: Density Plot for AGE
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.kdeplot(dataset_imputed[age_column], fill='true')
plt.title(f'Kernel Density Estimate (KDE) for {age_column}')
plt.xlabel(age_column)
plt.ylabel('Density')
plt.show()

#Code for part one-5: Density Plot for BMI
bmi_column = 'BMI'
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.kdeplot(dataset_imputed[bmi_column], fill='true')
plt.title(f'Kernel Density Estimate (KDE) for {bmi_column}')
plt.xlabel(bmi_column)
plt.ylabel('Density')
plt.show()

#Code for part one-6: Visualising Correlations
input_features = ['NPG', 'PGL', 'DIA', 'TSF', 'INS', 'BMI', 'DPF', 'AGE']
correlation_matrix = dataset_imputed[input_features + [diabetic_column]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap between Input Features and Diabetic')
plt.show()


#Code for part two-1: LR1
age_model_input = dataset_imputed.drop('AGE', axis=1)
age_model_output = dataset_imputed['AGE']
age_model_input_train, age_model_input_test, age_model_output_train, age_model_output_test = train_test_split(
    age_model_input, age_model_output, test_size=0.2, random_state=20
)
model_LR1 = LinearRegression()
model_LR1.fit(age_model_input_train, age_model_output_train)
age_predictions = model_LR1.predict(age_model_input_test)
mse = mean_squared_error(age_model_output_test, age_predictions)
r2 = r2_score(age_model_output_test, age_predictions)
print(f'Mean Squared Error for LR1: {mse}')
print(f'R-squared for LR1: {r2}')

plt.scatter(age_model_output_test, age_predictions, color='blue', label='Actual vs. Predicted')
plt.plot([min(age_model_output_test),
max(age_model_output_test)],
[min(age_model_output_test),
max(age_model_output_test)],
linestyle='--', color='red', label='Perfect Fit')
plt.title('Actual vs. Predicted AGE')
plt.xlabel('Actual AGE')
plt.ylabel('Predicted AGE')
plt.legend()
plt.show()

#Code for part two-2: LR2
age_model_input2 = dataset_imputed['NPG'].values.reshape(-1, 1)
age_model_output2 = dataset_imputed['AGE']
age_model_input_train2, age_model_input_test2, age_model_output_train2, age_model_output_test2 = train_test_split(
    age_model_input2, age_model_output2, test_size=0.2, random_state=20)
model_LR2 = LinearRegression()
model_LR2.fit(age_model_input_train2, age_model_output_train2)
age_predictions2 = model_LR2.predict(age_model_input_test2)
mse2 = mean_squared_error(age_model_output_test2, age_predictions2)
r2_2 = r2_score(age_model_output_test2, age_predictions2)
print(f'Mean Squared Error for LR2: {mse2}')
print(f'R-squared for LR2: {r2_2}')
plt.scatter(age_model_output_test2, age_predictions2, color='green', label='Actual vs. Predicted')
plt.plot([min(age_model_output_test2), max(age_model_output_test2)],
         [min(age_model_output_test2), max(age_model_output_test2)],
         linestyle='--', color='red', label='Perfect Fit')
plt.title('Actual vs. Predicted AGE')
plt.xlabel('Actual AGE')
plt.ylabel('Predicted AGE')
plt.legend()
plt.show()

#Code for part two-3: LR3
age_model_input3 = dataset_imputed[['NPG', 'DIA', 'PGL']]
age_model_output3 = dataset_imputed['AGE']
age_model_input_train3, age_model_input_test3, age_model_output_train3, age_model_output_test3 = train_test_split(
    age_model_input3, age_model_output3, test_size=0.2, random_state=20)
model_LR3 = LinearRegression()
model_LR3.fit(age_model_input_train3, age_model_output_train3)
age_predictions3 = model_LR3.predict(age_model_input_test3)
mse3 = mean_squared_error(age_model_output_test3, age_predictions3)
r2_3 = r2_score(age_model_output_test3, age_predictions3)
print(f'Mean Squared Error for LR3: {mse3}')
print(f'R-squared for LR3: {r2_3}')
plt.scatter(age_model_output_test3, age_predictions3, color='yellow', label='Actual vs. Predicted')
plt.plot([min(age_model_output_test3), max(age_model_output_test3)],
         [min(age_model_output_test3), max(age_model_output_test3)],
         linestyle='--', color='red', label='Perfect Fit')
plt.title('Actual vs. Predicted AGE')
plt.xlabel('Actual AGE')
plt.ylabel('Predicted AGE')
plt.legend()
plt.show()

#Code for part three: kNN to predict Diabetic
knn_input = dataset_imputed.drop('Diabetic', axis=1)
knn_output = dataset_imputed['Diabetic']
knn_input_train, knn_input_test, knn_output_train, knn_output_test = train_test_split(
    knn_input, knn_output, test_size=0.2, random_state=20
)
k = [6, 7, 9]
metrics = ['euclidean', 'manhattan', 'chebyshev']
for i in k:
    for m in metrics:
        knn = KNeighborsClassifier(n_neighbors=i, metric=m)
        knn.fit(knn_input_train, knn_output_train)
        knn_predictions = knn.predict(knn_input_test)
        precision = precision_score(knn_output_test, knn_predictions)
        recall = recall_score(knn_output_test, knn_predictions)
        f1 = f1_score(knn_output_test, knn_predictions)
        accuracy = accuracy_score(knn_output_test, knn_predictions)
        roc_auc = roc_auc_score(knn_output_test, knn_predictions)
        conf_matrix = confusion_matrix(knn_output_test, knn_predictions)
        print(
            f"K = {i}, Metrics = {m}, Precision = {precision:.4f}, "
            f"Recall = {recall:.4f}, F1 Score = {f1:.4f}, Accuracy = {accuracy:.4f}, ROC/AUC Score = {roc_auc:.4f}")
        fpr, tpr, _ = roc_curve(knn_output_test, knn_predictions)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        plt.figure()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
        print("\n")
