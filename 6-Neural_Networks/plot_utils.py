import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from utils import softmax


def plot_random_samples(X_train, y_train, classes=None, samples_per_class=10, shape=(28, 28), figsize=(5, 4), show_titles=False):
    num_classes = len(classes)

    plt.figure(figsize=figsize)
    for y, label in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].reshape(shape), cmap=plt.cm.Greys)
            plt.axis('off')
            if i == 0 and show_titles:
                plt.title(label)
    plt.show()
    
def plot_sample(X, y, idx=None, annot=False, shape=(28, 28)):
    if idx is None:
        idx = np.random.randint(0, X.shape[0] + 1)
        
    x = X[idx].reshape(shape)
    
    figsize = (shape[0] // 2, shape[1] // 2)
    plt.figure(figsize=figsize)
    sns.heatmap(x, annot=annot, cmap=plt.cm.Greys, cbar=False)
    plt.title(y[idx])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    

def plot_tsne(X, y):
    plt.figure(figsize=(14, 8))
    X_embedded = TSNE(n_components=2).fit_transform(X)
    
    cmap = plt.cm.Spectral
    for c in range(10):
        l = np.flatnonzero(c == y)
        plt.scatter(X_embedded[l, 0], X_embedded[l, 1], cmap=cmap, alpha=0.5, label="%d" %c)

    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best')
    plt.show()
    
    
def predict_and_plot(probs, x, y, mu, classes, shape=(28, 28)):    
    plt.figure(figsize=(10, 4))
    
    # plot the digit
    plt.subplot(1, 2, 1)
    x = x + mu
    plt.imshow(x.reshape(shape), interpolation='nearest', cmap=plt.cm.Greys)
    plt.xticks([])
    plt.yticks([])
    plt.title(classes[np.argmax(probs)])
    
    # plot top 5 predictions
    idx = np.argsort(probs)[5:]
    tick_label = [classes[i] for i in idx]
    color = ['g' if i == y else 'r' for i in idx]
    
    plt.subplot(1, 2, 2)
    plt.barh(range(5), width=probs[idx], tick_label=tick_label, color=color)
    plt.xlim(0, 1)
    plt.title("Top 5 predictions")
    plt.show()
    
    
def plot_confusion_matrix(y_true, y_pred, normalize=False, figsize=(12, 12)):
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    plt.figure(figsize=figsize)
    annot = cm/cm.sum(axis=1) if normalize else True
    sns.heatmap(cm, annot=annot, cmap=plt.cm.Blues, cbar=False)
    plt.title("Confusion Matrix")
    plt.show()