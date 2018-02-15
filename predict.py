import numpy as np
from DataProcessing import getTrainingData

X, Y = getTrainingData()

M = 9
D = X.shape[1]
K = len(set(Y))

W1= np.random.randn(D,M)
b1=np.zeros(M)
W2= np.random.randn(M,K)
b2= np.zeros(K)


def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def frwd(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)


P_Y_given_X = frwd(X,W1,b1,W2,b2)
predictions = np.argmax(P_Y_given_X, axis=1)


def classification_rate(Y,P):
    return np.mean(Y == P)


print "Score:", classification_rate(Y, predictions)


