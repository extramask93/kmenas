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
    iterations = 0
    while(True):
        iterations = iterations + 1
        distances = calculateDistances(X,centers,X.shape[1])
        indices = findClosestCenter(X,distances)
        centers,cdiv = calculateNewCenters(X,centers,indices)
        if(np.array_equal(oldcenters,centers)):
            break
        oldcenters = centers
    return (centers,cdiv,iterations)
def pca(x, TargetDimension) :
    C = np.cov(np.transpose(x))
    l, p = np.linalg.eig(C)
    indexes = np.argsort(l)[::-1]
    newX = np.dot(x, p[:, indexes[0 : TargetDimension]])
    return newX
def ReduceDim(X,targetDim):
    dimNr = X.shape[1]
    if(dimNr>targetDim):
        X = pca(X,TargetDimension = targetDim)
    temp = X
    X = np.zeros((X.shape[0],dimNr))
    X[:,:targetDim] = temp
    return X
def paint(X,cl,dim):
    X = ReduceDim(X,dim)
    dct = dict.fromkeys(cl,X[0])
    for i in range(0,X.shape[0]):
        dct[cl[i]] = np.vstack((dct[cl[i]],X[i]))
    for i in dct.keys():
        dct[i] = dct[i][1:]
    cc=cm.rainbow(np.linspace(0,1,len(dct.keys())))
    fig1 = plt.figure()
    if(dim == 3):
        ax1 = Axes3D(fig1)
        for i,col in zip(dct.keys(),cc):
            ax1.scatter(dct[i][:,0],dct[i][:,1],dct[i][:,2],c = col)
    else:
        for i,col in zip(dct.keys(),cc):
            plt.scatter(dct[i][:,0],dct[i][:,1],c = col)
    plt.show()
def paintWithCenter(C,XC,alll,dim):
    color=cm.rainbow(np.linspace(0,1,C.shape[0]))
    prevlen = len(C) #needed becouse we actualy want color differenly points from different centers
    if(dim == 3):
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(0,len(C)):
            indexa = prevlen
            indexb = indexa + len(XC[i])
            ax.scatter(alll[i][0],alll[i][1],alll[i][2], marker=r'$\clubsuit$',c = 'black',s = 300,zorder=10)
            ax.scatter(alll[indexa:indexb,0],alll[indexa:indexb,1],alll[indexa:indexb,2], '*',c = color[i])
            prevlen = indexb
    else:
        for i in range(0,len(C)):
            indexa = prevlen
            indexb = indexa + len(XC[i])
            plt.scatter(alll[i][0],alll[i][1], marker=r'$\clubsuit$',c = 'black',s = 300,zorder=10)
            plt.scatter(alll[indexa:indexb,0],alll[indexa:indexb,1], marker = '*',c = color[i])
            prevlen = indexb
    plt.show()
if __name__ == "__main__":
    dim = 2 # reduce dimensions to...
    X,cl = DataGen.LoadIris()
    #X,cl = DataGen.LoadPulsar() #will take a bit, set centers to 2
    #X,cl = DataGen.Generate4D(100,100)
    #X,foo = DataGen.Generate3D(200,80)
    #paint(X,cl,dim)
    [C,XC,ilosc_iteracji] = kmeans(X,nrOfCenters = 3)
    print(ilosc_iteracji)
    temp = C
    for i in range(0,len(XC)):
       temp = np.concatenate((temp,XC[i]))
    alll = ReduceDim(temp,dim)
    paintWithCenter(C,XC,alll,dim)

