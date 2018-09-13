import numpy as np
import pandas as pd

num_labels = 3

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

def normalize_inputs(X_train, X_test):
    """Normalize the training and test sets."""
    mu = np.mean(X_train, axis=1, keepdims=1)
    sigma2 = np.var(X_train, axis=1, keepdims=1)
    X_train -= mu
    X_train /= sigma2
    X_test -= mu
    X_test /= sigma2

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

def run_ova_logistic_regression(num_iterations, learning_rate, X, Y_orig):
    """Run one vs. all logistic regression on X and return computed weights."""
    n = X.shape[0]
    m = X.shape[1]
    
    # Initialize weight matrices.
    W = np.empty((num_labels, n, 1))
    W0 = np.empty((num_labels, 1))

    for label in range(num_labels):
        # Make new copy of Y_orig to prevent changing the original array,
        # then increment by 1 to make the range of its values [2, num_labels + 1].
        Y = Y_orig + 1
        # Change all target labels to 1 and others to 0.
        Y[Y != label + 2] = 0
        Y[Y == label + 2] = 1
        # Perform logistic regression as in binary classification.
        for iteration in range(num_iterations):
            W[label], W0[label] = run_logistic_regression(num_iterations, learning_rate, X, Y)
    return W, W0

def ova_predict(X, W, W0):
    """Make predictions on X based on the weights provided."""
    m = X.shape[1]
    highest_confidence = np.zeros((1, m))
    predictions = np.zeros((1, m))
    for label in range(num_labels):
        h = sigmoid(np.dot(W[label].T, X) + W0[label])
        for i in range(m):
            if highest_confidence[0, i] < h[0, i]:
                highest_confidence[0, i] = h[0, i]
                predictions[0, i] = label + 1
    return predictions

def run_ovo_logistic_regression(num_iterations, learning_rate, X_orig, Y_orig):
    """Run one vs. one logistic regression on X and return computed weights."""
    n = X_orig.shape[0]
    m = X_orig.shape[1]
    num_models = (num_labels * (num_labels - 1)) // 2

    # Initialize weight matrices.
    W = np.empty((num_models, n, 1))
    W0 = np.empty((num_models, 1))

    model_index = 0

    for label1 in range(num_labels):
        for label2 in range(label1 + 1, num_labels):
            # Form X and Y arrays.
            X1 = X_orig[:, Y_orig[0] == label1 + 1]
            Y1 = Y_orig[0, Y_orig[0] == label1 + 1]
            X2 = X_orig[:, Y_orig[0] == label2 + 1]
            Y2 = Y_orig[0, Y_orig[0] == label2 + 1]
            X = np.concatenate((X1, X2), axis=1)
            Y = np.concatenate((Y1, Y2))
            Y[Y == label1 + 1], Y[Y == label2 + 1] = 0, 1
            for iteration in range(num_iterations):
                W[model_index], W0[model_index] = run_logistic_regression(num_iterations, learning_rate, X, Y)
            model_index += 1
    return W, W0

def ovo_predict(X, W, W0):
    """Make predictions on X based on the weights provided."""
    m = X.shape[1]
    model_index = 0

    # Count votes for each label for each example.
    votes = np.zeros((num_labels, m))
    for label1 in range(num_labels):
        for label2 in range(label1 + 1, num_labels):
            h = sigmoid(np.dot(W[model_index].T, X) + W0[model_index])
            h[h < 0.5], h[h >= 0.5] = 0, 1
            votes[label1, h[0] == 0] += 1
            votes[label2, h[0] == 1] += 1
            model_index += 1

    # Tally votes and make predictions.
    max_votes = np.zeros((1, m))
    predictions = np.empty((1, m))
    for i in range(m):
        for label in range(num_labels):
            if max_votes[0, i] < votes[label, i]:
                max_votes[0, i] = votes[label, i]
                predictions[0, i] = label + 1
    return predictions

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = import_data('data4.xlsx')
    normalize_inputs(X_train, X_test)
    test_set_size = Y_test.shape[1]

    # Perform one vs. all logistic regression.
    W_ova, W0_ova = run_ova_logistic_regression(50, 0.01, X_train, Y_train)
    ova_predictions = ova_predict(X_test, W_ova, W0_ova)
    ova_class_accuracies = np.zeros(num_labels)
    ova_class_members = np.zeros(num_labels)
    ova_accuracy = 0
    for i in range(test_set_size):
        ova_class_members[Y_test[0][i] - 1] += 1
        if ova_predictions[0][i] == Y_test[0][i]:
            ova_class_accuracies[Y_test[0][i] - 1] += 1
            ova_accuracy += 1
    ova_class_accuracies /= ova_class_members
    ova_accuracy /= test_set_size
    print('One vs. All Accuracy : ', ova_accuracy)
    for label in range(num_labels):
        print('One vs. All Accuracy for label ', label + 1, ' : ', ova_class_accuracies[label])

    # Perform one vs. one logistic regression.
    W_ovo, W0_ovo = run_ovo_logistic_regression(50, 0.01, X_train, Y_train)
    ovo_predictions = ovo_predict(X_test, W_ovo, W0_ovo)
    ovo_class_accuracies = np.zeros(num_labels)
    ovo_class_members = np.zeros(num_labels)
    ovo_accuracy = 0
    for i in range(test_set_size):
        ovo_class_members[Y_test[0][i] - 1] += 1
        if ovo_predictions[0][i] == Y_test[0][i]:
            ovo_class_accuracies[Y_test[0][i] - 1] += 1
            ovo_accuracy += 1
    ovo_class_accuracies /= ovo_class_members
    ovo_accuracy /= test_set_size
    print('One vs. One Accuracy : ', ovo_accuracy)
    for label in range(num_labels):
        print('One vs. One Accuracy for label ', label + 1, ' : ', ovo_class_accuracies[label])
