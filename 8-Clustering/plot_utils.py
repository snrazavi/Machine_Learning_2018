import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances

import warnings


def plot_clusters(X, cluster_ids, centroids, title):
    K = centroids.shape[0]
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=50, c=cluster_ids, edgecolors='k', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', s=200, c=range(K), edgecolors='k')
    plt.title(title)
    plt.axis('equal')
    plt.show()


def plot_kmeans(X, initial_centroids, plot=True):
    m = X.shape[0]
    K = initial_centroids.shape[0]
    centroids = np.array(initial_centroids)
    old_cluster_ids = np.zeros((m,))
    
    while True:
        
        # find closest center to each data
        cluster_ids = np.array([np.argmin(np.linalg.norm(X[i] - centroids, axis=1)) for i in range(m)])
        
        if plot:
            plot_clusters(X, cluster_ids, centroids, 'Assigning Data to Clusters')

        # update cluster centers
        for k in range(K):
            centroids[k] = np.mean(X[cluster_ids==k], axis=0)
        
        if plot:
            plot_clusters(X, cluster_ids, centroids, 'Updating Cluster Centers')
        
        # stop
        if np.all(cluster_ids == old_cluster_ids):
            return cluster_ids, centroids
        else:
            old_cluster_ids = cluster_ids
            
            
def plot_kmeans_interactive(min_clusters=1, max_clusters=6):

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.60)

        def kmeans_step(frame, K):
            rng = np.random.RandomState(2)
            cluster_ids = np.zeros(X.shape[0])
            centroids = rng.randn(K, 2)

            nsteps = frame // 3

            for i in range(nsteps + 1):
                old_centroids = centroids
                
                if i < nsteps or frame % 3 > 0:
                    dist = euclidean_distances(X, centroids)
                    cluster_ids = dist.argmin(1)

                if i < nsteps or frame % 3 > 1:
                    centroids = np.array([X[cluster_ids==k].mean(0) for k in range(K)])
                    nans = np.isnan(centroids)
                    centroids[nans] = old_centroids[nans]
            
            # plot data
            c = cluster_ids if frame > 0 else 'w'
            plt.scatter(X[:, 0], X[:, 1], c=c, s=50, edgecolors='k', vmin=0, vmax=K - 1, alpha=0.6);
            
            # plot centroids
            plt.scatter(old_centroids[:, 0], old_centroids[:, 1], marker='o', c=range(K), s=200)
            plt.scatter(old_centroids[:, 0], old_centroids[:, 1], marker='o', c='black', s=50)

            # plot new centers if third frame
            if frame % 3 == 2:
                for i in range(K):
                    plt.annotate('', xy=centroids[i], xytext=old_centroids[i], 
                                 arrowprops=dict(arrowstyle='->', linewidth=1, color='k'))
                plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', c=range(K), s=200)
                plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', c='black', s=50)

            plt.xlim(-4, 4)
            plt.ylim(-2, 10)

            if frame % 3 == 1:
                plt.title("Assign data to nearest centroid", size=14)
            elif frame % 3 == 2:
                plt.title("Update centroids to cluster means", size=14)
            else:
                plt.title(" ", size=14)
                
                
    frame = widgets.IntSlider(value=0, min=0, max=50, step=1, desc='frame')
    K = widgets.IntSlider(value=4, min=min_clusters, max=max_clusters, desc='K')

    return widgets.interact(kmeans_step, frame=frame, K=K)
