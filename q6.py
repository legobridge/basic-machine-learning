import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def import_data(file_name):
    """Use pandas to import an excel file."""
    dataframe = pd.read_excel(file_name, header=None, dtype=object)
    return dataframe.values

def find_labels(X, centroids):
    """Assign labels to each example in X, given centroids."""
    m = X.shape[0]
    k = centroids.shape[0]
    labels = np.zeros(m, dtype=np.int64)
    dist = np.array([1e18] * m)
    for i in range(m):
        for j in range(k):
            my_dist = np.sum((X[i] - centroids[j]) ** 2)
            if my_dist < dist[i]:
                dist[i] = my_dist
                labels[i] = j
    return labels

def find_clusters(X, k, iterations):
    """Perform K-Means Clustering on X and return labels and centroids"""
    m = X.shape[0]
    n = X.shape[1]
    indices = np.random.choice(m, k, replace=False)
    centroids = X[indices]
    for iteration in range(iterations):
        labels = find_labels(X, centroids)
        new_centroids = np.zeros(centroids.shape)
        new_label_count = np.zeros((k, 1))
        for i in range(m):
            new_centroids[labels[i]] = new_centroids[labels[i]] + X[i]
            new_label_count[labels[i]] += 1
        centroids = new_centroids / new_label_count
    return labels, centroids

if __name__ == '__main__':
    X = import_data('data2.xlsx')

    labels, centroids = find_clusters(X, k=2, iterations=10)
    m = X.shape[0]

    # Prepare data to plot.
    plot_data = np.zeros((m, 5), dtype=object)
    plot_data[:, 0:3] = X[:, 0:3]
    # Normalize the 4th feature to [0, 1] for RGB plotting.
    plot_data[:, 3] = X[:, 3] / np.max(X[:, 3])
    # Assign '^' to label 0, 'o' to label 1.
    markers = np.array(['^', 'o'])
    plot_data[:, 4] = markers[labels]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(m):
        rgb_color = np.array([[1.0 - plot_data[i, 3], 0.1, 0.0 + plot_data[i, 3]]])
        ax.scatter(plot_data[i, 0], plot_data[i, 1], plot_data[i, 2],
                   c=rgb_color, marker=plot_data[i, 4])
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')

    plt.show()
