import math
import numpy as np
import pandas as pd

def import_data(file_name):
    """Import an excel file and divide it into training and test sets."""
    dataframe = pd.read_excel(file_name, header=None, dtype=object)
    data = dataframe.values
    m = data.shape[0]
    train_set_size = (6 * m) // 10 # Take 60% of data for training.
    test_set_size = m - train_set_size # Take the remaining for testing.
    np.random.shuffle(data)
    X_train, X_test, Y_train, Y_test = (data[0:train_set_size, 0:4],
                                        data[train_set_size:, 0:4],
                                        data[0:train_set_size, 4],
                                        data[train_set_size:, 4])
    X_train = X_train.T.astype(float)
    X_test = X_test.T.astype(float)
    Y_train.resize((1, train_set_size))
    Y_test.resize((1, test_set_size))
    return X_train, X_test, Y_train, Y_test

def gaussian(X_train, X_test):
    """Compute Gaussian probability given X_train and X_test."""
    n = X_test.shape[0]
    m = X_test.shape[1]

    # Calculate parameter values required for the Gaussian model.
    mu = np.mean(X_train, axis=1, keepdims=True)
    cov = np.cov(X_train)
    err = X_test - mu
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)

    # Calculate constant left half of the distribution.
    l = 1 / ((((2*math.pi) ** n) * cov_det) ** 0.5)

    # Calculate right half of the distribution for each example.
    r = np.empty(m)
    for i in range(m):
        r[i] = np.exp((-0.5) * np.dot(err[:, i].T, np.dot(cov_inv, err[:, i])))
    return l * r

def predict(X_train, Y_train, X_test):
    """Make predictions on X_test based on the training data."""
    train_set_size = X_train.shape[1]
    test_set_size = X_test.shape[1]

    # Separate out training data according to labels.
    X_train_sep = [X_train[:, Y_train[0] == 1], X_train[:, Y_train[0] == 2]]
    # Compute priors.
    freq_labels = np.array([np.count_nonzero(Y_train == 1), np.count_nonzero(Y_train == 2)])
    priors = freq_labels / train_set_size
    
    # Calculate probibilities p(X|y) for both models.
    probs_x_given_y = np.array([gaussian(X_train_sep[i], X_test) for i in range(2)])

    # Make predictions based on LRT rule.
    predictions = np.empty((1, test_set_size))
    for i in range(test_set_size):
        if probs_x_given_y[0, i] / probs_x_given_y[1, i] >= priors[1] / priors[0]:
            predictions[0, i] = 1
        else:
            predictions[0, i] = 2
    return predictions

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = import_data('../../datasets/data3.xlsx')

    predictions = predict(X_train, Y_train, X_test)
    test_set_size = Y_test.shape[1]
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(test_set_size):
        if predictions[0, i] == 2 and Y_test[0, i] == 2:
            tp += 1
        if predictions[0, i] == 2 and Y_test[0, i] == 1:
            fp += 1
        if predictions[0, i] == 1 and Y_test[0, i] == 1:
            tn += 1
        if predictions[0, i] == 1 and Y_test[0, i] == 2:
            fn += 1
    print(tp, fp, tn, fn)
    sensitivity = tp / (tp + fn)
    print("Sensitivity : ", sensitivity)
    specificity = tn / (tn + fp)
    print("Specificity : ", specificity)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Accuracy : ", accuracy)

