import math
import numpy as np
import pandas as pd

num_labels = 3

def import_data(file_name):
    """Import an excel file and divide it into training and test sets."""
    dataframe = pd.read_excel(file_name, header=None, dtype=object)
    data = dataframe.values
    m = data.shape[0]
    train_set_size = (7 * m) // 10 # Take 70% of data for training.
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
    X_train_sep = [X_train[:, Y_train[0] == label] for label in range(1, num_labels + 1)]
    
    # Compute priors.
    freq_labels = np.array([np.count_nonzero(Y_train == label) for label in range(1, num_labels + 1)])
    priors = freq_labels / train_set_size
    priors.resize((num_labels, 1))

    # Calculate probibilities p(X|y) and p(X|y)*p(y) for all models.
    probs_x_given_y = np.array([gaussian(X_train_sep[i], X_test) for i in range(num_labels)])
    posterior = probs_x_given_y * priors

    # Make predictions based on MAP rule.
    predictions = np.argmax(posterior, axis=0) + 1
    predictions.resize((1, test_set_size))
    return predictions

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = import_data('../../datasets/data4.xlsx')
    test_set_size = Y_test.shape[1]

    # Perform MAP predictions.
    predictions = predict(X_train, Y_train, X_test)
    class_accuracies = np.zeros(num_labels)
    class_members = np.zeros(num_labels)
    accuracy = 0
    for i in range(test_set_size):
        class_members[Y_test[0, i] - 1] += 1
        if predictions[0, i] == Y_test[0, i]:
            class_accuracies[Y_test[0, i] - 1] += 1
            accuracy += 1
    class_accuracies /= class_members
    accuracy /= test_set_size
    print('Accuracy : ', accuracy)
    for label in range(num_labels):
        print('Accuracy for label ', label + 1, ' : ', class_accuracies[label])