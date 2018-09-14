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

    file_name = '../../datasets/data.xlsx'
    dataframe = pd.read_excel(file_name, header=None, dtype=object)
    data = dataframe.values
    X, Y = data[:, 0:2], data[:, 2]
    X = X.T
    Y = Y.T
    return X, Y


'''
    Runs Batch Gradient Descent for Ridge Regression.
    Returns:
    W0, W - The optimized weights
    costs - A record of the costs at all iterations
'''
def run_batch_gd(num_iterations, learning_rate, lambd, X, Y):

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
            cost += 0.5*np.square(err) + lambd*(np.abs(W0) + np.linalg.norm(W, ord=1))
            dW0 += err + 0.5*lambd*np.sign(W0)
            dW += err * np.array([[X[0][i]], [X[1][i]]]) + 0.5*lambd*np.sign(W)
        W0 = W0 - (learning_rate * dW0) / m
        W = W - (learning_rate * dW) / m
        costs[iteration] = cost / m

    return W0, W, costs


'''
    Runs Stochastic Gradient Descent for Ridge Regression.
    Returns:
    W0, W - The optimized weights
    costs - A record of the costs at all iterations
'''
def run_stochastic_gd(num_iterations, learning_rate, lambd, X, Y):

    m = X.shape[1]
    W0 = np.zeros(1)
    W = np.array([[0.3], [0]])

    costs = np.zeros(num_iterations)

    for iteration in range(num_iterations):
        cost = 0
        for i in range(m):
            prediction = W0 + W[0]*X[0][i] + W[1]*X[1][i]
            err = prediction - Y[i]
            cost += 0.5*np.square(err) + lambd*(np.abs(W0) + np.linalg.norm(W, ord=1))
            dW0 = err + 0.5*lambd*np.sign(W0)
            dW = err * np.array([[X[0][i]], [X[1][i]]]) + 0.5*lambd*np.sign(W)
            W0 = W0 - learning_rate * dW0
            W = W - learning_rate * dW
        costs[iteration] = cost / m

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
    W0, W, costs = run_batch_gd(2000, 0.000001, 100, X, Y)
    plot(costs, 'Batch Gradient Descent')
    print("Final weight vector (Batch GD) : ", [W0[0], W[0][0], W[1][0]])
    W0, W, costs = run_stochastic_gd(200, 0.0000001, 100, X, Y)
    plot(costs, 'Stochastic Gradient Descent')
    print("Final weight vector (Stochastic GD) : ", [W0[0], W[0][0], W[1][0]])