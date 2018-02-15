import numpy as np
from scipy.special import expit
from DataProcessing import getTrainingData, getTestingData


X,Y= getTrainingData()
#X,Y = getTestingData()

def sigmoid(inputs):
    return expit(inputs)

def sigmoidDerivative(inputs):
    ex = np.exp(-inputs)
    return ex / (1 + ex)**2


def softmax(A):
    #print('A',A)
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y


def feedForward(X,W1,b1,W2,b2,training):
    #print('W2',W2)
    #print('b2',b2)
    #outputFirst=X.dot(W1)+b1
    #print('outputFirst',outputFirst)
    #Z=np.tanh(outputFirst)
    Z= 1/1+np.exp(-X.dot(W1)-b1)

    if training:
        m2 = np.random.binomial(1, 0.5, size=Z.shape)
    else:
        m2 = 0.5
    Z *= m2

    outputSecond = Z.dot(W2) + b2

    #Z=relu(outputFirst)
    #print('Z',Z)

    #print('outputSecond *!!!',outputSecond)
    P=softmax(outputSecond)

    return P,Z,m2


def classification_rate(Y, P):
    return np.mean(Y == P)

def relu(x):
    return np.maximum(x, 0, x)

def relu_derivate(x):
    return np.where(x > 0, 1, 0)
