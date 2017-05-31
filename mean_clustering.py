# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:12:12 2015

@author: chad
"""

from __future__ import division
import random as random
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#
#from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
#import plotly.plotly as py
#import plotly.graph_objs as go

from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py
__date__ = "2011-11-17 Nov denis"

#init_notebook_mode()

# Functions
# Read Data
def readStrain(filename):
    rownum = 0
    print("Reading Strain Data Here")
    # Try reading and printing file
    with open(filename) as csvfile:
        data = csv.reader(csvfile)
        for r in data:
            #print(rownum)
            if rownum == 0:
                mean_data = np.array([r])
                #print("-----")
            else:
                mean_data = np.concatenate((mean_data,np.array([r])))
                #print("+++++")
            rownum += 1
    return(mean_data)

# K-means Functions
# kmeans.py using any of the 20-odd metrics in scipy.spatial.distance
# kmeanssample 2 pass, first sample sqrt(N)

    # X sparse, any cdist metric: real app ?
    # centres get dense rapidly, metrics in high dim hit distance whiteout
    # vs unsupervised / semi-supervised svm

#...............................................................................
#def kmeans( X, centres, delta=.0001, maxiter=100, metric="euclidean", p=2, verbose=1 ):
def kmeans( X, centres, delta, maxiter, metric, verbose, p=2 ):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    if verbose:
        print "kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s" % (
            X.shape, centres.shape, delta, maxiter, metric)
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = cdist_sparse( X, centres, metric=metric, p=p )  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print "kmeans: av |X - nearest centre| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print "kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc)
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans: cluster 50 % radius", r50 #.astype(int)
        print "kmeans: cluster 90 % radius", r90 #.astype(int)
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances,r50,r90

#...............................................................................
def kmeanssample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centres
    """
        # merge w kmeans ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centres = randomsample( X, int(k) )
    samplecentres = kmeans( Xsample, pass1centres, **kwargs )[0]
    return kmeans( X, samplecentres, **kwargs )

def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    return d

def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]

def nearestcentres( X, centres, metric="euclidean", p=2 ):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()

#...............................................................................
class Kmeans:
    """ km = Kmeans( X, k= or centres=, ... )
        in: either initial centres= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centres, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centres[jcentre]
                J indexes e.g. X[J], classes[J]
    """
    def __init__( self, X, k=0, centres=None, nsample=0, **kwargs ):
        self.X = X
        if centres is None:
            self.centres, self.Xtocentre, self.distances = kmeanssample(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centres, self.Xtocentre, self.distances = kmeans(
                X, centres, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)

#...............................................................................

# Main Program

mean_data = readStrain("D:\Projects\Stephenson Centre\Proof of Concept\Data\mean_data_norm.csv")
#mean_data = readStrain("D:\Projects\Stephenson Centre\Proof of Concept\Data\mean_data3.csv")

#print("After readStrain()")
#print(mean_data)

# Data to cluster

#print(mean_data[[0],0:27])
name_data = (mean_data[[0],2:])
data = np.delete(mean_data,0,0)

#Anomalous Patient - removed
data = np.delete(data,87,0)

patients = (data[:,[0]])
data = np.delete(data,0,1)
data=data.astype(float)

diabetic = (data[:,[0]])
data = np.delete(data,0,1)

x_coord = 9
y_coord = 10

#data = (data[:,[x_coord,y_coord]])
#data = (data[:,[0,1,2,3,4,5,6,7,8,9,10,22,23,24]])
#data = (data[:,:])
#data = (data[:,[0,1,2,3,4,5,7,8,9,10,22,23,24]]) # e... + end
#print name_data[:,[0,1,2,3,4,5,7,8,9,10,22,23,24]]
data = (data[:,[0,1,2,3,4,5,7,8,9]]) # e...
name_data = name_data[:,[0,1,2,3,4,5,7,8,9]]
#data = (data[:,[11,12,13,14,15,16,17,18,19,20,21]]) # tt...
#name_data = name_data[:,[11,12,13,14,15,16,17,18,19,20,21]]

non_dia_index = np.where(diabetic == 0)
non_dia_data = data[non_dia_index[0],:]
non_dia_pat = patients[non_dia_index[0],:]
#print "Non-diabetic Patients: ", non_dia_pat 
dia_index = np.where(diabetic == 1)
dia_data = data[dia_index[0],:]
dia_pat = patients[dia_index[0],:]
#print "Diabetic Patients: ", dia_pat

# --------------------------

if __name__ == "__main__":
    import sys
    from time import time

#    N = 91
#    dim = 0
    ncluster = 4
    kmsample = 0  # 0: random centres, > 0: kmeanssample
    kmdelta = .0001
    kmiter = 100
#    metric = "cityblock"  # "chebyshev" = max, "cityblock" L1,  Lqmetric, "euclidean"
#    metric = "chebyshev"
#    metric = "cityblock"
#    metric = Lqmetric
    metric = "euclidean"

    seed = 1

    exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...
    np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True )
    np.random.seed(seed)
    random.seed(seed)

    print "ncluster %d  kmsample %d  metric %s" % (ncluster, kmsample, metric)

#    X = data
#    X_pat = patients
#    X = non_dia_data
#    X_pat = non_dia_pat
    X = dia_data
    X_pat = dia_pat
#   X = np.random.exponential( size=(N,dim) )
        # cf scikits-learn datasets/
    t0 = time()
    if kmsample > 0:
        centres, xtoc, dist,r50,r90 = kmeanssample( X, ncluster, nsample=kmsample,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    else:
        randomcentres = randomsample( X, ncluster )
        centres, xtoc, dist,r50,r90 = kmeans( X, randomcentres,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
            
#    print "%.0f msec" % ((time() - t0) * 1000)
    

    index = 0
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0
    for i in xtoc:
        if i == 0:
#            print "patient %s - dist %f " % (patients[index],dist[index])
            if t1 == 0:
                new_dataset1 = (X[index:index+1,])
                patient1 = (X_pat[index,[0]])
                t1 = 1
            else:
                new_dataset1 = np.append(new_dataset1,X[index:index+1,],axis=0)
                patient1 = np.append(patient1, X_pat[index,[0]],axis=0)
        elif i == 1:
#            print "cluster 2"
            if t2 == 0:
                new_dataset2 = (X[index:index+1,])
                patient2 = (X_pat[index,[0]])
                t2 = 1
            else:
                new_dataset2 = np.append(new_dataset2,X[index:index+1,],axis=0)
                patient2 = np.append(patient2, X_pat[index,[0]],axis=0)
        elif i == 2:
            if t3 == 0:
                new_dataset3 = (X[index:index+1,])
                patient3 = (X_pat[index,[0]])
                t3 = 1
            else:
                new_dataset3 = np.append(new_dataset3,X[index:index+1,],axis=0)
                patient3 = np.append(patient3, X_pat[index,[0]],axis=0)
        elif i == 3:
            if t4 == 0:
                new_dataset4 = (X[index:index+1,])
                patient4 = (X_pat[index,[0]])
                t4 = 1
            else:
                new_dataset4 = np.append(new_dataset4,X[index:index+1,],axis=0)           
                patient4 = np.append(patient4, X_pat[index,[0]],axis=0)
        elif i == 4:
            if t5 == 0:
                new_dataset5 = (X[index:index+1,])
                patient5 = (X_pat[index,[0]])
                t5 = 1
            else:
                new_dataset5 = np.append(new_dataset5,X[index:index+1,],axis=0)            
                patient5 = np.append(patient5, X_pat[index,[0]],axis=0)
        else:
            print "Not in cluster"

        index += 1
    
    cx = 2
    cy = 7
    cz = 5
#    plt.subplot(2,1,1)
    plt.plot(new_dataset1[:,[cx]],new_dataset1[:,[cy]],'bs')
    if ncluster >= 2:
        plt.plot(new_dataset2[:,[cx]],new_dataset2[:,[cy]],'r^')
    if ncluster >= 3: 
        plt.plot(new_dataset3[:,[cx]],new_dataset3[:,[cy]],'gx')
    if ncluster >= 4:
        plt.plot(new_dataset4[:,[cx]],new_dataset4[:,[cy]],'k>')
    if ncluster >= 5:
        plt.plot(new_dataset5[:,[cx]],new_dataset5[:,[cy]],'y^')
    plt.plot(centres[:,[cx]],centres[:,[cy]],'go',ms=10)
    plt.axis([0,1,0,1])
    plt.plot([0,1],[0,1],'r--') 
    plt.axis([0.2,1,0.2,1])
    plt.xlabel(name_data[0,cx])
    plt.ylabel(name_data[0,cy])
    plt.title("Normalised - All Patients - 3 clusters in 9D")
    
#    fig = plt.figure()
##    plt.subplot(2,1,2)
#    plt.plot(centres[[0],:],centres[[1],:],'gs')
#    plt.plot(centres[[0],:],centres[[2],:],'r^')
#    plt.plot(centres[[1],:],centres[[2],:],'ys')
#    if ncluster >=4:
#        plt.plot(centres[[0],:],centres[[3],:],'bo')
#        plt.plot(centres[[1],:],centres[[3],:],'c^')
#        plt.plot(centres[[2],:],centres[[3],:],'kx')
#
#    plt.plot([0,1],[0,1],'r--') 
#    plt.axis([0.2,1,0.2,1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new_dataset1[:,[cx]], new_dataset1[:,[cy]], new_dataset1[:,[cz]], zdir='z', s=20, c='b')
    ax.scatter(new_dataset2[:,[cx]], new_dataset2[:,[cy]], new_dataset2[:,[cz]], zdir='z', s=20, c='r')
    ax.scatter(new_dataset3[:,[cx]], new_dataset3[:,[cy]], new_dataset3[:,[cz]], zdir='z', s=20, c='g')
    ax.scatter(new_dataset4[:,[cx]], new_dataset4[:,[cy]], new_dataset4[:,[cz]], zdir='z', s=20, c='k')
    ax.scatter(new_dataset5[:,[cx]], new_dataset5[:,[cy]], new_dataset5[:,[cz]], zdir='z', s=20, c='y')
    ax.scatter(centres[:,[cx]],centres[:,[cy]], centres[:,[cz]], zdir='z', s=100, c='k')

    new_data_std1 = np.std(new_dataset1[:,:], axis=0)
    new_data_per1 = new_data_std1/centres*100
#    print centres[[0],[cx]], new_data_std1, new_data_std1/centres[[0],[cx]]*100
    new_data_std2 = np.std(new_dataset2[:,:], axis=0)
    new_data_per2 = new_data_std2/centres*100
#    print centres[[1],[cx]], new_data_std2, new_data_std2/centres[[1],[cx]]*100
    new_data_std3 = np.std(new_dataset3[:,:], axis=0)
    new_data_per3 = new_data_std3/centres*100
#    print centres[[2],[cx]], new_data_std3, new_data_std3/centres[[2],[cx]]*100
    new_data_std4 = np.std(new_dataset4, axis=0)
#    print centres[[3],[cx]], new_data_std4 , new_data_std4/centres[[3],[cx]]*100

    new_data_std5 = np.std(new_dataset5, axis=0)
 #   print centres[[4],[cx]], new_data_std5 , new_data_std5/centres[[4],[cx]]*100
    
#    print "Cluster 1:"
#    print patient1
#    
#    print "Cluster 2:"
#    print patient2
#
#
#    print "Cluster 3:"
#    print patient3
#    
    