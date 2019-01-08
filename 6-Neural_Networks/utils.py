import numpy as np


def accuracy(y_pred, y_true):
    return 100. * np.mean(y_pred == y_true)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_loss(scores, y, mode='train'):
    m = scores.shape[0]
    probs = softmax(scores)
    loss = -np.sum(np.log(probs[range(m), y])) / m
    
    if mode != 'train':
        return loss
    
    # backward
    dscores = probs
    dscores[range(m), y] -= 1.0
    dscores /= m
    
    return loss, dscores