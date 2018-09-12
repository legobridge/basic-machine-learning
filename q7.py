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
    X_train = X_train.T
    X_test = X_test.T
    Y_train.resize((1, train_set_size))
    Y_test.resize((1, test_set_size))
    return X_train, X_test, Y_train, Y_test

def sigmoid(Z):
    """Apply the sigmoid function to the input Z."""
    return 1 / (1 + np.exp(-Z.astype(float)))

def run_logistic_regression(num_iterations, learning_rate, X, Y):
    """Run logistic regression on X and return computed weights."""
    n = X.shape[0]
    m = X.shape[1]
    
    # Initialize weights.
    W = np.random.randn(n, 1)
    W0 = 0

    for iteration in range(num_iterations):
        h = sigmoid(np.dot(W.T, X) + W0)
        cost = -(1 / m) * np.sum((Y*np.log(h) + (1 - Y)*np.log(1 - h)))
        dW = (1 / m) * np.dot(X, (h - Y).T)
        W = W - learning_rate*dW
        dW0 = (1 / m) * np.sum(h - Y)
        W0 = W0 - learning_rate*dW0
    return W, W0

def predict(X, W, W0):
    """Make predictions on X based on the weights provided."""
    h = sigmoid(np.dot(W.T, X) + W0)
    predictions = h
    predictions[h < 0.5], predictions[h >= 0.5] = 0, 1
    return predictions + 1

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = import_data('data3.xlsx')

    W, W0 = run_logistic_regression(100, 0.01, X_train, Y_train - 1)
    predictions = predict(X_test, W, W0)
    test_set_size = Y_test.shape[1]
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(test_set_size):
        if predictions[0][i] == 2 and Y_test[0][i] == 2:
            tp += 1
        if predictions[0][i] == 2 and Y_test[0][i] == 1:
            fp += 1
        if predictions[0][i] == 1 and Y_test[0][i] == 1:
            tn += 1
        if predictions[0][i] == 1 and Y_test[0][i] == 2:
            fn += 1
    print(tp, fp, tn, fn)
    sensitivity = tp / (tp + fn)
    print("Sensitivity : ", sensitivity)
    specificity = tn / (tn + fp)
    print("Specificity : ", specificity)
    accuracy = tp + tn / (tp + fp + tn + fn)
    print("Accuracy : ", accuracy)

