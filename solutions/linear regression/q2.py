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

def run_stochastic_gd(num_iterations, learning_rate, X, Y):
    """Run stochastic gradient descent and return the weights and costs."""
    m = X.shape[1]
    W0 = np.zeros(1)
    W = np.array([[0.3], [0]])
    costs = np.zeros(num_iterations)
    for iteration in range(num_iterations):
        cost = 0
        for i in range(m):
            prediction = W0 + W[0]*X[0][i] + W[1]*X[1][i]
            err = prediction - Y[i]
            cost += 0.5 * np.square(err)
            dW0 = err
            dW = err * np.array([[X[0][i]], [X[1][i]]])
            W0 = W0 - learning_rate * dW0
            W = W - learning_rate * dW
        costs[iteration] = cost / m
    return W0, W, costs

def plot(costs):
    """Plot the graph of Cost vs. iterations."""
    fig1, ax1 = plt.subplots()
    ax1.plot(costs)
    ax1.set(xlabel='Iterations', ylabel='Cost', title='Stochastic Gradient Descent')
    ax1.grid()
    plt.show()

if __name__ == '__main__':

    X, Y = import_data()
    W0, W, costs = run_stochastic_gd(200, 0.0000001, X, Y)
    plot(costs)
    print("Final weight vector : ", [W0[0], W[0][0], W[1][0]])