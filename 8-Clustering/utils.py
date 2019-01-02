import numpy as np


def find_closest_center(x, centroids):
    """ Find center in centers which has minimum distance to d
    
        Arguments:
            x: data point in 2d space (1d numpy array)
            centers: cluster centers (numpy array of shape (K, 2))
            
        Returns:
            the index of closest cluster center
    """
    return np.argmin(np.linalg.norm(x - centroids, axis=1))


def kMeans(X, initial_centroids, max_iters=10):
    K = initial_centroids.shape[0]
    centroids = np.array(initial_centroids)
    cluster_ids = None

    for i in range(max_iters):
        cluster_ids = np.array([find_closest_center(X[i], centroids) for i in range(X.shape[0])])
        for k in range(K):
            centroids[k] = np.mean(X[cluster_ids==k], axis=0)

    return cluster_ids, centroids


def compress(img, K=16, max_iters=10):
    X = img / 255.0
    img_size = img.shape

    X = X.reshape(img_size[0] * img_size[1], 3)  # to 3d matrix

    # run kmeans
    initial_centroids = np.random.permutation(X)[:K]
    cluster_ids, centroids = kMeans(X, initial_centroids, max_iters)
        
    # mapping the centroids back to compressed image,
    m = X.shape[0]
    X_compressed = np.zeros_like(X)

    for i in range(0, m):
        k = cluster_ids[i]
        X_compressed[i] = centroids[k]

    X_compressed = X_compressed.reshape(img_size[0], img_size[1], 3)
    
    return X_compressed