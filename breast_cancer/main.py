import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold

# Breast cancer dataset for classification


# load dataset
data = load_breast_cancer()
X = pd.DataFrame(data=data.data, columns=data.feature_names)
Y = pd.DataFrame(data=data.target)
print("Original data:")
print(X)
print()
print("Original results:")
print(Y)
print()
Y = numpy.int32(Y)


# preprocessing data
print("1. Standard Scaler")
print("2. Min Max Scaler")
print("3. Max Abs Scaler")
normalizer = int(input("Choose a normalizer: "))

if normalizer == 1:
    preprocessing_pipeline = make_pipeline(StandardScaler())
    preprocessing_pipeline.fit(X)
    X_normalized = preprocessing_pipeline.transform(X)
    print("Normalized data1:")
    print(X_normalized)
    print()

elif normalizer == 2:
    preprocessing_pipeline = make_pipeline(MinMaxScaler())
    preprocessing_pipeline.fit(X)
    X_normalized = preprocessing_pipeline.transform(X)
    print("Normalized data2:")
    print(X_normalized)
    print()

elif normalizer == 3:
    preprocessing_pipeline = make_pipeline(MaxAbsScaler())
    preprocessing_pipeline.fit(X)
    X_normalized = preprocessing_pipeline.transform(X)
    print("Normalized data3:")
    print(X_normalized)
    print()

else:
    print("Wrong input!")
    exit(0)


# feature selection
print("1. Select K Best features")
print("2. Mutual info Classifier")
print("3. Tree-based feature selection")
feature_selector = int(input("Choose a feature selector: "))

k = 5
if feature_selector == 1:
    X_final = SelectKBest(f_classif, k=k).fit_transform(X_normalized, np.ravel(Y))
    feature_indexes1 = []
    for i in range(len(X_final[0, :])):
        feature_indexes1.append(np.where(X_normalized[0, :] == X_final[0, i])[0][0])
    print("Final features from SelectKBest:")
    print(X.columns[feature_indexes1])
    print(X_final)

elif feature_selector == 2:
    correlaton_coefs = mutual_info_classif(X_normalized, np.ravel(Y), n_neighbors=k)
    correlaton_coefs_indexes = np.argsort(correlaton_coefs)
    X_final = X_normalized[:, correlaton_coefs_indexes[-k:]]
    feature_indexes2 = []
    for i in range(len(X_final[0, :])):
        feature_indexes2.append(np.where(X_normalized[0, :] == X_final[0, i])[0][0])
    print("Final features from Mutual info Classifier:")
    print(X.columns[feature_indexes2])
    print(X_final)

elif feature_selector == 3:
    correlaton_coefs = ExtraTreesClassifier()
    correlaton_coefs = correlaton_coefs.fit(X_normalized, np.ravel(Y))
    model = SelectFromModel(correlaton_coefs, max_features=k)
    X_final = model.transform(X_normalized)
    feature_indexes3 = []
    for i in range(len(X_final[0, :])):
        feature_indexes3.append(np.where(X_normalized[0, :] == X_final[0, i])[0][0])
    print("Final features from Tree-based feature selection:")
    print(X.columns[feature_indexes3])
    print(X_final)
else:
    print("Wrong input!")
    exit(0)

# division of data into training and test sets and classification
print("1. 10-Fold Cross Validation + kNN Model")
print("2. Single Random Division + kNN Model")
print("3. kNN Model")
print("4. kNN Model")
data_classificator = int(input("Choose a data classificator: "))

k = 10
n = 5
kNN_model = KNeighborsClassifier(n_neighbors=n)

if data_classificator == 1:
    mean_accuracy = 0
    k_fold = KFold(n_splits=k, random_state=None, shuffle=False)
    for i, (train_indexes, test_indexes) in enumerate(k_fold.split(X_final)):
        kNN_model.fit(X_final[train_indexes, :], Y[train_indexes])
        Y_predicted = kNN_model.predict(X_final[test_indexes])
        print("kNN Model:")
        for i in range(len(Y_predicted)):
            if Y_predicted[i] == Y[test_indexes[i]]:
                print(f'{i}. predicted: {Y_predicted[i]} RESULT: PASSED')
            else:
                print(f'{i}. predicted: {Y_predicted[i]} RESULT: FAILED')

        accuracy = float(sum((y_p == y) for y_p, y in zip(Y_predicted, Y[test_indexes])) / len(Y_predicted) * 100)
        mean_accuracy += accuracy
        print(f"Accuracy = {round(accuracy, 4)}%")
    mean_accuracy /= k
    print(f"Mean Accuracy = {round(mean_accuracy, 4)}%")

elif data_classificator == 2:
    X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size=0.1, random_state=None)
    kNN_model.fit(X_train, Y_train)
    Y_predicted = kNN_model.predict(X_test)
    print("kNN Model:")
    for i in range(len(Y_predicted)):
        if Y_predicted[i] == Y_test[i]:
            print(f'{i}. predicted: {Y_predicted[i]} RESULT: PASSED')
        else:
            print(f'{i}. predicted: {Y_predicted[i]} RESULT: FAILED')

    accuracy = float(sum((y_p == y) for y_p, y in zip(Y_predicted, Y_test)) / len(Y_predicted) * 100)
    print(f"Accuracy = {round(accuracy, 4)}%")

elif data_classificator == 3:
    kNN_model.fit(X_final, Y)
    Y_predicted = kNN_model.predict(X_final)
    print("kNN Model:")
    for i in range(len(Y)):
        if Y_predicted[i] == Y[i]:
            print(f'{i}. predicted: {Y_predicted[i]} RESULT: PASSED')
        else:
            print(f'{i}. predicted: {Y_predicted[i]} RESULT: FAILED')

    accuracy = float(sum((y_p == y) for y_p, y in zip(Y_predicted, Y)) / len(Y) * 100)
    print(f"Accuracy = {round(accuracy, 4)}%")

elif data_classificator == 4:
    kNN_model.fit(X_final, Y)
    Y_predicted = kNN_model.predict(X_final)
    print("kNN Model:")
    for i in range(len(Y)):
        if Y_predicted[i] == Y[i]:
            print(f'{i}. predicted: {Y_predicted[i]} RESULT: PASSED')
        else:
            print(f'{i}. predicted: {Y_predicted[i]} RESULT: FAILED')

    accuracy = float(sum((y_p == y) for y_p, y in zip(Y_predicted, Y)) / len(Y) * 100)
    print(f"Accuracy = {round(accuracy, 4)}%")

else:
    print("Wrong input!")
    exit(0)