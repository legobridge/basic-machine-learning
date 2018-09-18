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

def run_batch_gd(num_iterations, learning_rate, X, Y):
    """Run batch gradient descent and return the weights and costs."""
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
            cost += 0.5 * np.square(err)
            dW0 += err
            dW += err * np.array([[X[0][i]], [X[1][i]]])
        W0 = W0 - (learning_rate * dW0) / m
        W = W - (learning_rate * dW) / m
        costs[iteration] = cost / m
    return W0, W, costs

def plot(W0, costs, X, Y):
    """Plot the two graphs."""
    m = X.shape[1]

    # Plot the graph of Cost vs. iterations.
    fig1, ax1 = plt.subplots()
    ax1.plot(costs)
    ax1.set(xlabel='Iterations', ylabel='Cost', title='Batch Gradient Descent')
    ax1.set_ylim((0, 120))
    ax1.grid()

    # Plot the graph of Cost vs. W1 and W2.
    num_samples = 100
    W1s = np.linspace(-1.00, 1.00, num_samples)
    W2s = np.linspace(-0.02, 0.04, num_samples)
    W1s, W2s = np.meshgrid(W1s, W2s)
    costs = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            W_temp = np.array([[W1s[0, j]], [W2s[i, 0]]])
            predictions_temp = W0 + np.dot(W_temp.T, X)
            err_temp = predictions_temp - Y
            costs[i, j] = (0.5  / m) * (np.sum(np.square(err_temp)))

    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    surf = ax2.plot_surface(W1s, W2s, costs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.set_xlabel('W1', fontdict=dict(weight='heavy'))
    ax2.set_ylabel('W2', fontdict=dict(weight='heavy'))
    ax2.set_zlabel('Cost', fontdict=dict(weight='heavy'))
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax2.zaxis.set_major_formatter(FormatStrFormatter('%0.0e'))

    plt.show()
    

if __name__ == '__main__':

    X, Y = import_data()
    W0, W, costs = run_batch_gd(2000, 0.000001, X, Y)
    plot(W0, costs, X, Y)
    print("Final weight vector : ", [W0[0], W[0][0], W[1][0]])