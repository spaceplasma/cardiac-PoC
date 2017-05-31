# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 13:39:45 2016

@author: bryant
"""

from __future__ import division
import random as random
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from time import time

#
#from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
#import plotly.plotly as py
#import plotly.graph_objs as go

from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py

# Functions
# Read Data
def readData(filename):
    rownum = 0
#    print("Reading Data Here")
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

#--------------------------------------------------------------------

t0 = time() # Calculate run time

full_data = readData("D:\Projects\Stephenson Centre\Proof of Concept\Data\expanded_data.csv")

patients = full_data[1:,[0]]
unused_data = full_data[:,[1,2,3,4,5,18,31,32,56,58,59,60,61,62,65,67,68,69,70,71,74,75,76,77,78,79,80,81,82,111]]
data = np.delete(full_data,[0,1,2,3,4,5,18,31,32,56,58,59,60,61,62,65,67,68,69,70,71,74,75,76,77,78,79,80,81,82,111],1)
data_column_headers = data[[0],:]
data = np.delete(data,[0],0)

data[np.where(data == "")] = -99

data=data.astype(float)

ncluster = 3
kmsample = 0  # 0: random centres, > 0: kmeanssample
kmdelta = .0001
kmiter = 100
#    metric = "cityblock"  # "chebyshev" = max, "cityblock" L1,  Lqmetric, "euclidean"
#    metric = "chebyshev"
#    metric = "cityblock"
#    metric = Lqmetric
metric = "euclidean"

exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...

seed = 1

np.random.seed(seed)
random.seed(seed)

#Correlation Matrix for data

data_cont = np.delete(data,[1,2,3,7,8,9,10,11,12,13,18,19,23,27,28,29,30,45,46,47,51,52,56,57,58,59,65,66,67,68,70,71,72,73],1)
data_cont_headers = np.delete(data_column_headers,[1,2,3,7,8,9,10,11,12,13,18,19,23,27,28,29,30,45,46,47,51,52,56,57,58,59,65,66,67,68,70,71,72,73],1)
#normalize the dataset
data_cont = data_cont/np.amax(data_cont,axis=0)
data_corr_mat = np.corrcoef(data_cont,rowvar=0)
plt.imshow(data_corr_mat)

#define data set to use

## lge
#dataset = data[:,81:128]
#dataset_headers = data_column_headers[:,81:128]

## Troponin, QRSWidth,QTc,SV,EDV, CO
dataset = data[:,[15,17,79,24,167]]
dataset_headers = data_column_headers[:,[15,17,79,24,167]]

## Age, HR, QRSWidth, QTc,EDV,ESV,LVMass,LVEF,SV,BPs,BPd,BMI
#dataset = data[:,[0,14,15,17,80,24,25,26,79,162,163,169]]
#dataset_headers = data_column_headers[:,[0,14,15,17,80,24,25,26,79,162,163,169]]

## Age, HR, QRSWidth, QTc,EDV,ESV,LVMass,LVEF,SV,BPs,BPd,BSA, LGE5SDs
#dataset = data[:,[0,14,15,17,80,24,25,26,79,162,163,166,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128]]
#dataset_headers = data_column_headers[:,[0,14,15,17,80,24,25,26,79,162,163,166,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128]]

# Find Rows containing -99 and eliminate them
indices = np.where(dataset == -99)
unq_rows = np.unique(indices[0])
#print unq_rows
dataset=np.delete(dataset,unq_rows,0)
patients = np.delete(patients,unq_rows,0)

## Remove Troponin issues
#indices = np.where(dataset[:,0] >= 90)
#unq_rows = np.unique(indices[0])
##print unq_rows
#dataset=np.delete(dataset,unq_rows,0)
#patients = np.delete(patients,unq_rows,0)


X = dataset
X_patients = patients

# Normalize the dataset
#X = X/np.amax(X,axis=0)

# Try feature scaling
X = (X-np.mean(X,axis=0))/(np.amax(X,axis=0)-np.amin(X,axis=0))

print "ncluster %d  kmsample %d  metric %s" % (ncluster, kmsample, metric)

if kmsample > 0:
    centres, xtoc, dist,r50,r90 = kmeanssample( X, ncluster, nsample=kmsample,
                                               delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
else:
    randomcentres = randomsample( X, ncluster )
    centres, xtoc, dist,r50,r90 = kmeans( X, randomcentres,
                                         delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )

cluster1_data = X[np.where(xtoc == 0)]
cluster2_data = X[np.where(xtoc == 1)]
cluster3_data = X[np.where(xtoc == 2)]
#cluster4_data = X[np.where(xtoc == 3)]
#cluster5_data = X[np.where(xtoc == 4)]

cluster1_patients = patients[np.where(xtoc == 0)]
cluster2_patients = patients[np.where(xtoc == 1)]
cluster3_patients = patients[np.where(xtoc == 2)]
#cluster4_patients = patients[np.where(xtoc == 3)]
#cluster5_patients = patients[np.where(xtoc == 4)]

cluster1_std = np.std(cluster1_data, axis=0)
cluster2_std = np.std(cluster2_data, axis=0)
cluster3_std = np.std(cluster3_data, axis=0)
#cluster4_std = np.std(cluster4_data, axis=0)
#cluster5_std = np.std(cluster5_data, axis=0)


# LAST LINE PLEASE
print "Run Time: %.0f msec" % ((time() - t0) * 1000)
