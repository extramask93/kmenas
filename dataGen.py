import numpy as np
from scipy.io import arff
import pandas as pd
class DataGen:
    def LoadIris(ReduceDimensions = 0):
        data = arff.loadarff("iris.arff")
        D1 = pd.DataFrame(data[0])
        columns = D1.columns.tolist()
        X = np.array(D1)
        X = X.astype('U13')
        D1 = X[:, -1]
        X = X[:,:-1-ReduceDimensions]
        X = X.astype(np.float)
        return X,D1
    def LoadBodyFat(ReduceDimensions = 0):
        data = arff.loadarff("bodyfat.arff")
        D1 = pd.DataFrame(data[0])
        columns = D1.columns.tolist()
        X = np.array(D1)
        D1 = X[:, -1]
        X = X.astype('U13')
        if(ReduceDimensions):
            X = X[:,:-1-ReduceDimensions]
        X = X.astype(np.float)
        return X,D1
    def Generate2D(nrOfSamples,nrOfNoiseSamples):
        X = np.random.randint(10, 45, (nrOfSamples, 2))
        Y = np.random.randint(55, 70, (nrOfSamples, 2))
        Z = np.random.randint(150, 190, (nrOfSamples, 2))
        noise = np.random.randint(0,190,(nrOfNoiseSamples,2))
        points = np.vstack((X,Y,Z))
        pointsNoised = np.vstack((X, Y,Z, noise))
        return pointsNoised,points
    def Generate3D(nrOfSamples,nrOfNoiseSamples):
        X = np.random.randint(10, 45, (nrOfSamples, 3))
        Y = np.random.randint(55, 70, (nrOfSamples, 3))
        Z = np.random.randint(80, 115, (nrOfSamples, 3))
        noise = np.random.randint(0, 105, (nrOfNoiseSamples, 3))
        points = np.vstack((X, Y,Z))
        pointsNoised = np.vstack((X, Y, Z, noise))
        return pointsNoised,points
