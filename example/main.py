import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error

# load data
X, Y = load_breast_cancer(return_X_y=True)
print(X)

# normalization
normalize_pipeline = make_pipeline(StandardScaler())
X_normalized = normalize_pipeline.fit_transform(X)
#X_normalized = X
print(X_normalized)

# kNN feature selection
feature_correlation_coef = mutual_info_classif(X_normalized, Y, discrete_features='auto', n_neighbors=10)
print("Features correlation coefficients: ", feature_correlation_coef)

# how much features to select
features_number = 3
sorted_coef_indexes = np.argsort(feature_correlation_coef)
X_final = X[:, sorted_coef_indexes[-3:]]

# training
model1 = LogisticRegression()
model1.fit(X_final, Y)
Y_predicted = model1.predict(X_final)

# evaluation
print(f'real results: {Y}')
print(f'predicted results: {Y_predicted}')
MSE = mean_squared_error(Y, Y_predicted, squared=True)
print(f'MSE = {MSE}')
accuracy = sum(y == y_p for y, y_p in zip(Y, Y_predicted)) / len(Y_predicted) * 100
print(f'Accuracy = {accuracy}')
print(list(zip(Y, Y_predicted)))


