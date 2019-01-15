import numpy as np
class DataGen:
    def Generate2D(nrOfSamples,nrOfNoiseSamples):
        X = np.random.randint(10, 45, (nrOfSamples, 2))
        Y = np.random.randint(55, 70, (nrOfSamples, 2))
        noise = np.random.randint(0,100,(nrOfNoiseSamples,2))
        points = np.vstack((X,Y))
        pointsNoised = np.vstack((X, Y, noise))
        return pointsNoised,points
    def Generate3D(nrOfSamples,nrOfNoiseSamples):
        X = np.random.randint(10, 45, (nrOfSamples, 3))
        Y = np.random.randint(55, 70, (nrOfSamples, 3))
        Z = np.random.randint(80, 115, (nrOfSamples, 3))
        noise = np.random.randint(0, 105, (nrOfNoiseSamples, 3))
        points = np.vstack((X, Y,Z))
        pointsNoised = np.vstack((X, Y, Z, noise))
        return pointsNoised,points
