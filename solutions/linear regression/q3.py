import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd

def import_data():
    """Import data from an excel file."""
    file_name = '../../datasets/data.xlsx'
    dataframe = pd.read_excel(file_name, header=None, dtype=object)
    data = dataframe.values
    X, Y = data[:, 0:2], data[:, 2]
    X = X.T
    Y = Y.T
    return X, Y

def run_batch_gd(num_iterations, learning_rate, lambd, X, Y):
    """Run batch gradient descent with ridge regression."""
    m = X.shape[1]
    W0 = np.zeros(1)
    W = np.array([[0.3], [0]])
    costs = np.zeros(num_iterations)
    for iteration in range(num_iterations):
        cost = 0
        dW0 = 0
        dW = 0
        for i in range(m):
            prediction = W0 + W[0]*X[0][i] + W[1]*X[1][i]
            err = prediction - Y[i]
            cost += 0.5 * (np.square(err) + lambd*(W0**2 + np.linalg.norm(W)**2))
            dW0 += err + lambd*W0
            dW += err * np.array([[X[0][i]], [X[1][i]]]) + lambd*W
        W0 = W0 - (learning_rate * dW0) / m
        W = W - (learning_rate * dW) / m
        costs[iteration] = cost / m
    return W0, W, costs

def run_stochastic_gd(num_iterations, learning_rate, lambd, X, Y):
    """Run stochastic gradient descent with ridge regression."""
    m = X.shape[1]
    W0 = np.zeros(1)
    W = np.array([[0.3], [0]])
    costs = np.zeros(num_iterations)
    for iteration in range(num_iterations):
        cost = 0
        for i in range(m):
            prediction = W0 + W[0]*X[0][i] + W[1]*X[1][i]
            err = prediction - Y[i]
            cost += 0.5 * (np.square(err) + lambd*(W0**2 + np.linalg.norm(W)**2))
            dW0 = err + lambd*W0
            dW = err * np.array([[X[0][i]], [X[1][i]]]) + lambd*W
            W0 = W0 - learning_rate * dW0
            W = W - learning_rate * dW
        costs[iteration] = cost / m
    return W0, W, costs

def plot(costs, graph_title):
    """Plot the graph of Cost vs. iterations."""
    fig1, ax1 = plt.subplots()
    ax1.plot(costs)
    ax1.set(xlabel='Iterations', ylabel='Cost', title=graph_title)
    ax1.set_ylim((0, 120))
    ax1.grid()
    plt.show()

if __name__ == '__main__':

    X, Y = import_data()
    W0, W, costs = run_batch_gd(2000, 0.000001, 1000, X, Y)
    plot(costs, 'Batch Gradient Descent')
    print("Final weight vector (Batch GD) : ", [W0[0], W[0][0], W[1][0]])
    W0, W, costs = run_stochastic_gd(200, 0.0000001, 1000, X, Y)
    plot(costs, 'Stochastic Gradient Descent')
    print("Final weight vector (Stochastic GD) : ", [W0[0], W[0][0], W[1][0]])