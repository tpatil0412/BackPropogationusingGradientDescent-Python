import numpy as np
from DataProcessing import getTrainingData, getTestingData, get_data
from scipy.special import expit
from FeedForward import feedForward


X, Y = get_data()

Xtrain, Ytrain= getTrainingData()
Xtest, Ytest = getTestingData()

#print Xtrain.shape, Xtest.shape
#print Ytrain.shape,Ytest.shape

M = 10
D = X.shape[1]
K = Y.shape[1]

#print D,K

W1= np.random.randn(D,M)
b1=np.zeros(M)
W2= np.random.randn(M,K)
b2= np.zeros(K)


def cross_entropy(T, Y):
    return -np.mean(T * np.log(Y))


def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)


def accuracy(predictions, labels):
    #print('100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)',100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)))
    print(' Correct :', np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)))
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

#Gradient Desecent


learning_rate = 0.00001
for i in range(15000):
    Ptrain, Ztrain, m2train=feedForward(Xtrain,W1,b1,W2,b2,True)
    error_train= cross_entropy(Ytrain,Ptrain)
    # update W1 with L2 Regularization

    W2 -= learning_rate*((Ztrain.T.dot(Ptrain - Ytrain))+0.1*W2)
    b2 -= learning_rate*(Ptrain - Ytrain).sum(axis=0)
    dZ = (Ptrain - Ytrain).dot(W2.T) * Ztrain*(1 - Ztrain)*m2train
    W1 -= learning_rate*((Xtrain.T.dot(dZ))+0.1*W1)
    b1 -= learning_rate*dZ.sum(axis=0)

    if (i%1000==0):
        print('Iteration:', i)
        print('classification rate for training data : ',accuracy(Ptrain,Ytrain))
        print('******************************')



for i in xrange(15000):
    Ptest, Ztest, m2test = feedForward(Xtest, W1, b1, W2, b2, False)
    error_test = cross_entropy(Ytest, Ptest)
    if(i%1000 == 0):
        print("Iteration: ", i)
        print('classification rate for testing data : ', accuracy(Ptest, Ytest))

print('Number of nodes at Hidden Layer :',M)
print('Iterations ',20000)
print('Learning Rate',learning_rate)
print('Final accuracy for training data :',accuracy(Ptrain,Ytrain))
print('Final accuracy for testing data :',accuracy(Ptest,Ytest))

