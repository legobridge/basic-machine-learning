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
	Runs Stochastic Gradient Descent.
	Returns:
	W0, W - The optimized weights
	costs - A record of the costs at all iterations
'''
def run_stochastic_gd(num_iterations, learning_rate, X, Y):

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


''' 
	Plots two graphs:
	1) Cost vs. num_iterations
	2) Surface graph of Cost vs. W1 and W2
'''
def plot(W0, costs, X, Y):

	m = X.shape[1]

	fig1, ax1 = plt.subplots()
	ax1.plot(costs)
	ax1.set(xlabel='Iterations', ylabel='Cost', title='Stochastic Gradient Descent')
	ax1.grid()

	fig2 = plt.figure()
	ax2 = fig2.gca(projection='3d')
	W1s = np.linspace(-0.4, 0.4, 100)
	W2s = np.linspace(-0.1, 0.1, 100)
	costs = np.zeros((100, 100))

	for i in range(100):
		for j in range(100):
			W_temp = np.array([[W1s[i]], [W2s[j]]])
			predictions_temp = W0 + np.dot(W_temp.T, X)
			err_temp = predictions_temp - Y
			costs[i][j] = (0.5  / m) * (np.sum(np.square(err_temp)))

	W1s, W2s = np.meshgrid(W1s, W2s)
	surf = ax2.plot_surface(W1s, W2s, costs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax2.set_xlabel('W1', fontdict=dict(weight='heavy'))
	ax2.set_ylabel('W2', fontdict=dict(weight='heavy'))
	ax2.set_zlabel('Cost', fontdict=dict(weight='heavy'))
	fig2.colorbar(surf, shrink=0.5, aspect=5)
	ax2.xaxis.set_major_locator(LinearLocator(10))
	ax2.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax2.yaxis.set_major_locator(LinearLocator(10))
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax2.zaxis.set_major_locator(LinearLocator(10))
	ax2.zaxis.set_major_formatter(FormatStrFormatter('%0.0e'))

	plt.show()


if __name__ == '__main__':

	X, Y = import_data()
	W0, W, costs = run_stochastic_gd(200, 0.0000001, X, Y)
	plot(W0, costs, X, Y)
	print("Final weight vector : ", [W0[0], W[0][0], W[1][0]])