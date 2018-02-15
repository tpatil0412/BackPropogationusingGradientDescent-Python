import sys

sys.path.append('../ann_logistic_extra')

from DataProcessing import get_data, getTrainingData, getTestingData
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle



# get the data
#X,Y = get_data()
Xtrain, Ytrain = getTrainingData()
Xtest,Ytest = getTestingData()


# split into train and test
Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
Xtest, Ytest = shuffle(Xtest, Ytest)


# create the neural network

model = MLPClassifier(hidden_layer_sizes=(7,), max_iter=15000)


# train the neural network
model.fit(Xtrain, Ytrain)


# print the train and test accuracy
train_accuracy = model.score(Xtrain, Ytrain)

test_accuracy = model.score(Xtest, Ytest)

print ('Sci-Kit Model')
print('Number of nodes at Hidden Layer: 8')
print('Iterations used: 15,000')
print "Train accuracy:", train_accuracy, "Test accuracy:", test_accuracy