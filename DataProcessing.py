import pandas as pd
import numpy as np
from sklearn.utils import shuffle


df = pd.read_csv('adult.csv')
df.head()
df = df[(df.workclass != ' ?') & (df.occupation != ' ?') & (df['native-country'] != ' ?')]

df = df.drop('fnlwgt', axis = 1)
df = df.drop('education', axis=1)
df = df.drop('relationship', axis=1)
df = df.drop('race', axis=1)
df = df.drop('capital-loss', axis=1)
df = df.drop('capital-gain', axis=1)
df = df.reset_index(drop=True)
#df.set_index('age', inplace=True)


#df = df.set_index(df.index)
#df = df.drop('index', axis=1)
final_df = df



df = pd.get_dummies(df, columns = ['sex', 'workclass', 'marital-status', 'occupation', 'native-country'])

input = df.loc[:, df.columns != 'class']
output = df.loc[:, 'class']



input,output = shuffle(input,output)

print input.shape

dataInput = input.as_matrix()

#print dataOutput

#print(dataInput.shape)
# normalize columns 1 and 2
dataInput[:, 0] = (dataInput[:, 0] - dataInput[:, 0].mean()) / dataInput[:, 0].std()
dataInput[:, 1] = (dataInput[:, 1] - dataInput[:, 1].mean()) / dataInput[:, 1].std()
dataInput[:, 2] = (dataInput[:, 2] - dataInput[:, 2].mean()) / dataInput[:, 2].std()


def get_data():
    final_X = dataInput           #input data with one hot encoding
    final_Y = output.as_matrix()  #output data without one hot encoding

    return final_X,final_Y

output = pd.get_dummies(output, columns = ['class'])
output.head()
dataOutput= output.as_matrix()

#print output

dataInput = input.as_matrix()

#print dataOutput

#print(dataInput.shape)
# normalize columns 1 and 2
# dataInput[:, 0] = (dataInput[:, 0] - dataInput[:, 0].mean()) / dataInput[:, 0].std()
# dataInput[:, 2] = (dataInput[:, 2] - dataInput[:, 2].mean()) / dataInput[:, 2].std()
# dataInput[:, 6] = (dataInput[:, 6] - dataInput[:, 6].mean()) / dataInput[:, 6].std()


trainingDataCount= int(0.70*(dataInput.shape[0]))



def getTrainingData():
    trainingInput= dataInput[:trainingDataCount]
    trainingOutPut=dataOutput[:trainingDataCount]
    return trainingInput, trainingOutPut


def getTestingData():
    testingInput= dataInput[trainingDataCount:]
    testingOutPut=dataOutput[trainingDataCount:]
    return testingInput,testingOutPut