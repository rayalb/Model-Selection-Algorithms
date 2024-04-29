#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:09:55 2023

@author: ray
"""

'''
    Density-Based Spatial Clustering of Application with Noise (DBSCAN).
    
    Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 
    "A density-based algorithm for discovering clusters in large 
    spatial databases with noise". 
    In Proceedings of the Second International Conference on Knowledge 
    Discovery and Data Mining 
    1996.
    
    https://dl.acm.org/doi/10.5555/3001460.3001507

'''


import numpy as np
from scipy.spatial.distance import cdist 


def RegionQuery(ii, D, epsilon):
    Neighbors = np.nonzero(D[ii,:]<=epsilon)
    return Neighbors[0]

def ExpandCluster(ii, visited, Neighbors, C, IDX, D, epsilon, minpts):
    IDX[ii] = C
    kk = 0
    while True:
        jj = Neighbors[kk]
        if ~visited[jj]:
            visited[jj] = True
            Neighbors2 = RegionQuery(jj, D, epsilon)
            if Neighbors2.size>=minpts:
                Neighbors = np.concatenate((Neighbors, Neighbors2))                
                
        if IDX[jj] == 0:
            IDX[jj] = C
            
        kk+=1
        
        if kk >= Neighbors.size:
            break
        
    return IDX, Neighbors, visited
    

def DBSCAN(X, epsilon, minpts, row_index):
    C = 0
    n = X.shape[0]
    IDX = np.zeros(n)
    
    D = cdist(X, X)
    
    very_large = 1000*epsilon
    for i1 in np.arange(1,n):
        for i2 in np.arange(i1-1):
            if row_index[i1] == row_index[i2]:
                D[i1,i2] = very_large
                D[i2,i1] = very_large
    
    visited = np.zeros(n, dtype=bool)
    isnoise = np.zeros(n, dtype=bool)
    
    for ii in range(n):
        if ~(visited[ii]):
            visited[ii] = True
            Neighbors = RegionQuery(ii, D, epsilon)
            if Neighbors.size < minpts:
                isnoise[ii] = True
            else:
                C = C + 1
                IDX, Neighbors, visited = ExpandCluster(ii, visited, Neighbors, 
                                                        C, IDX, D, epsilon, minpts)
    
    return IDX, isnoise
                
    
def find_cluster(points, epsilon = 5e-2, minpts = 3, marker = '.'):
    '''
      Perform a cluster analysis of points in 2D using DBSCAN

    Parameters
    ----------
    points : TYPE
        nxm matrix points represented by complex number.
    epsilon : double, optional
        DBSCAN epsilon parameter. The default is 5e-2.
    minpts : int, optional
        DBSCAN minpts parameters. The default is 3.

    Returns
    -------
    xy : Points in clustters.
    cluster_index : cluster index. 

    '''
    nrows, mcols = points.shape
    row_index = np.zeros((nrows,mcols))
    
    for rr in range(nrows):
        row_index[rr,:] = rr
        
    row_index = row_index.flatten('F')
    xy = np.zeros((nrows*mcols,2))
    xy[:,0] = points.real.flatten('F')
    xy[:,1] = points.imag.flatten('F')
    
    cluster_index, _ = DBSCAN(xy, epsilon, minpts, row_index)
    '''
    ncluster = int(max(cluster_index))
    for cc in range(ncluster):
        subset = (cluster_index == cc+1)
        xyc = xy[subset,:]
        plt.plot(xyc[:,0], xyc[:,1], marker)
    '''
    return xy, cluster_index

