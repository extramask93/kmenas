#k means
#1.wybor k centr
#2.wyznaczenie odleglosci punktow od centr
#3.przypisanie punktow do centr
#4.zaktualizowanie polozenia centr
#5.powtarzaj 2-4 tak dlugo, jak polozenie centr sie zmienia
import numpy as np
from scipy.io import arff
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from dataGen import DataGen

def selectCenters(X,k):
    Centers = np.zeros([k,X.shape[1]])
    for i in range(0,k):
        index = random.randint(0,len(X))
        Centers[i] = X[index]
    return Centers
def calculateDistances(X,C,p):
    D = np.zeros([len(X),len(C)])
    for i in range(0,C.shape[0]):#for each center
        suma = 0
        for j in range(0,X.shape[0]): #for each point
            suma = np.sum(np.power(np.abs(X[j,:]-C[i,:]),p))
            suma = np.power(suma,1/p)
            D[j][i] = suma
    return D #D[pointIndex][CenterIndex]
def findClosestCenter(X,D):
    best = np.zeros([X.shape[0],1])
    for i in range(0,X.shape[0]):
        b = np.argmin(D[i])
        best[i] = b
    return best #returns closest centers for all of the points
def calculateNewCenters(X,C,indices):
    #for each dimension of points associated with given center
    #calculate median and it is our new center
    newC = np.zeros(C.shape)
    cdiv = {} #maps points to the center
    for i in range(0,C.shape[0]):
        cnt = 0
        for j in range(0,X.shape[0]):
            if(indices[j] == i):
                for d in range(0,C.shape[1]):
                    newC[i][d] = newC[i][d] + X[j][d]
                cnt = cnt + 1
                if(i not in cdiv):
                    cdiv[i] = X[j]
                else:
                    cdiv[i] = np.vstack((cdiv[i], X[j]))              
        newC[i] = np.divide(newC[i],cnt) #median
    return newC,cdiv
def kmeans(X,nrOfCenters):
    centers = selectCenters(X,nrOfCenters)
    oldcenters = centers
    while(True):
        distances = calculateDistances(X,centers,X.shape[1])
        indices = findClosestCenter(X,distances)
        centers,cdiv = calculateNewCenters(X,centers,indices)
        if(np.array_equal(oldcenters,centers)):
            break;
        oldcenters = centers
    return (centers,cdiv)
def pca(x, TargetDimension) :
    C = np.cov(np.transpose(x))
    l, p = np.linalg.eig(C)
    indexes = np.argsort(l)[::-1]
    newX = np.dot(x, p[:, indexes[0 : TargetDimension]])
    return newX

if __name__ == "__main__":
    #X = DataGen.LoadIris()
    #X = DataGen.LoadBodyFat()
    #X,foo = DataGen.Generate3D(100,10)
    X,foo = DataGen.Generate2D(100,10)
    [C,XC] = kmeans(X,nrOfCenters = 3)
    #XC is a dictionary where keys are indexes of centers - > example C[0] = [1,2,3,4] then XC.at(0) has 2D aggregation of points [[..][..][..]] 
    #belonging to the center
    #concatenate centers and points for statistics incorporated in pca algorithm
    temp = C
    for i in range(0,len(XC)):
        temp = np.concatenate((temp,XC[i]))
    if(C.shape[1]>3):
        alll = pca(temp,TargetDimension = 3)
    else:
        alll = np.zeros((temp.shape[0],3))
        alll[:,:temp.shape[1]] = temp
    #colors
    color=cm.rainbow(np.linspace(0,1,C.shape[0]))
    prevlen = len(C) #needed becouse we actualy want color differenly points from different centers
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(0,len(C)):
        indexa = prevlen
        indexb = indexa + len(XC[i])
        ax.scatter(alll[i][0],alll[i][1],alll[i][2], marker='^',c = color[i],s = 100)
        ax.scatter(alll[indexa:indexb,0],alll[indexa:indexb,1],alll[indexa:indexb,2], '*',c = color[i])
        prevlen = indexb 
    plt.show()
