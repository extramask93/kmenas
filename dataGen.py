import numpy as np
from scipy.io import arff
import pandas as pd
class DataGen:
    def LoadIris():
        data = arff.loadarff("iris.arff")
        D1 = pd.DataFrame(data[0])
        columns = D1.columns.tolist()
        X = np.array(D1)
        X = X.astype('U13')
        D1 = X[:, -1]
        X = X[:,:-1]
        X = X.astype(np.float)
        return X,D1
    def LoadPulsar():
        data = arff.loadarff("HTRU_2.arff")
        D1 = pd.DataFrame(data[0])
        columns = D1.columns.tolist()
        X = np.array(D1)
        D1 = X[:, -1]
        X = X.astype('U13')
        X = X[:,:-1]
        X = X.astype(np.float)
        return X,D1
    def Generate4D(nrOfSamples,nrOfNoiseSamples):
        X = np.random.randint(5, 88, (nrOfSamples, 4))
        Y = np.random.randint(300, 400, (nrOfSamples, 4))
        Z = np.random.randint(-100, -9, (nrOfSamples, 4))
        noise = np.random.randint(-150,400,(nrOfNoiseSamples,4))
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
