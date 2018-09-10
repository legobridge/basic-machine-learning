import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd

'''
    Uses pandas to import the excel file.
    Returns:
    X - Feature matrix
    Y - Label vector
'''
def import_data():

    file_name = 'data.xlsx'
    dataframe = pd.read_excel(file_name, header=None, dtype=object)
    data = dataframe.values
    X, Y = data[:, 0:2], data[:, 2]
    X = X.T
    Y = Y.T
    return X, Y


'''
    Runs a vectorized version of Batch Gradient Descent.
    Returns:
    W0, W - The optimized weights
    costs - A record of the costs at all iterations
'''
def run_batch_gd(num_iterations, learning_rate, X, Y):

    m = X.shape[1]
    W0 = np.zeros(1)
    W = np.array([[0.3], [0]])

    costs = np.zeros(num_iterations)

    for iteration in range(num_iterations):
        predictions = W0 + np.dot(W.T, X)
        err = predictions - Y
        cost = (0.5  / m) * (np.sum(np.square(err)))
        costs[iteration] = cost
        dW = np.dot(err, X.T)
        dW.resize((2, 1))
        W0 = W0 - learning_rate * np.sum(err) / m
        W = W - learning_rate * dW / m

    return W0, W, costs


''' 
    Plots Cost vs. num_iterations
'''
def plot(costs, graph_title):

    fig1, ax1 = plt.subplots()
    ax1.plot(costs)
    ax1.set(xlabel='Iterations', ylabel='Cost', title=graph_title)
    ax1.set_ylim((0, 120))
    ax1.grid()
    plt.show()


if __name__ == '__main__':

    X, Y = import_data()
    W0, W, costs = run_batch_gd(2000, 0.000001, X, Y)
    plot(costs, 'Vectorized Batch Gradient Descent')
    print("Final weight vector : ", [W0[0], W[0][0], W[1][0]])