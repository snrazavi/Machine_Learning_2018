import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_svc_decision_function(clf, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
        
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([[xi, yj]])
    
    # plot the margins
    ax.contour(X, Y, P, colors='k', 
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    
def make_spiral(density=1, max_radius=6.5, c=0):
    """ Generate spiral dataset.
    
    Arguments:
        density (int)  : Density of the points
        maxRadius (float) : Maximum radius of the spiral
        c (int) : Class of this spiral
    
    Returns:
        array: Return spiral data and its class
    """
    
    # Spirals data and labels
    data, labels = [], []

    # Number of interior data points to generate
    N = 96 * density 

    # Generate points
    for i in range(0, N):
        angle = (i * math.pi) / (16 * density)
        # Radius is the maximum radius * the fraction of iterations left
        radius = max_radius * ((104 * density) - i) / (104 * density)

        # Get x and y coordinates
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        data.append([x, y])
        labels.append([c])

    return data, labels

def make_spirals(density=1, max_radius=6.5):
    """ Generate two class spiral dataset.

    Arguments:
        density (int)  : Density of the points
        maxRadius (float) : Maximum radius of the spiral
    Returns:
        array: Return spirals data and its class
    """
    data , labels = [], []

    # First spirals data and class
    data1, labels1 = make_spiral(density, max_radius)

    # Construct complete two spirals dataset
    for d in data1:
        data.append(d)  # First spirals coordinate
        data.append([-d[0], -d[1]])  # Second spirals coordinate

    # Construct complete two spirals classes
    for lbl in labels1:
        labels.append(lbl)  # First spirals class
        labels.append([1])  # Second spirals class

    return np.array(data), np.array(labels).ravel()