import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


def load_data(fname='data/ex2data1.txt', include_bias=True):
    """ Load data from a text file and save them in a 2D matrix.
    
    Arguments:
        - fname: the name of the text file containing data
        - include_bias: add a columns of ones to the input features.
        
    Returns:
        - X: the input features, a 2d matrix of shape (m, n+1)
        - y: the target vector, a 1d vector of shape (m,)
    """
    data = np.genfromtxt(fname, delimiter=',')
    m = data.shape[0]
    X = data[:, :-1].reshape(m, -1)   # get all but the last column
    y = data[:, -1]                   # get the last column
    X = np.hstack((np.ones((m, 1)), X)) if include_bias else X
    return X, y


def map_features(x1, x2, degree=1):
    """ Create polynomial features up to input degree for the input features x1 and x2.
    
    Arguments:
        - degree: the degree of polynomial features
        - x1: input feature, a 1d vector
        - x2: input feature, a 1d vector
        
    Returns:
        - A 2d matrix containing polynomial features up to input degree.
    """
    X = np.ones((x1.shape[0], 1))
    
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            f = (x1 ** (i - j)) * (x2 ** j)
            X = np.hstack((X, f.reshape(-1, 1)))
    return X


def plot_data(X, y, xlabel='Exam 1 score', ylabel='Exam 2 score', labels=['y = 0', 'y = 1'], alpha=0.5):
    pos = X[y == 1]
    neg = X[y == 0]

    plt.scatter(neg[:, 1], neg[:, 2], s=50, c='r', marker='x', alpha=alpha, label=labels[0])
    plt.scatter(pos[:, 1], pos[:, 2], s=50, c='b', marker='o', alpha=alpha, edgecolors='k', label=labels[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    
def plot_decision_boundary(theta, X, y, xlabel='Exam 1 score', ylabel='Exam 2 score', degree=1, title=None):
    """ Draw a binary decision boundary.
    """
    plt.figure(figsize=(12, 8))
    x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x2 = np.linspace(X[:, 2].min(), X[:, 2].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_poly = map_features(X1.ravel(), X2.ravel(), degree)
    probabilities = expit(X_poly @ theta.T)
    c = ['r' if p < 0.5 else 'b' for p in probabilities]
    plt.scatter(X_poly[:, 1], X_poly[:, 2], s=5, marker='o', c=c, alpha=0.2)
    plot_data(X, y, xlabel, ylabel, alpha=0.7)
    if title:
        plt.title(title)
    plt.show()