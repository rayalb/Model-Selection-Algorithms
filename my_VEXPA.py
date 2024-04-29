#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:44:23 2023

@author: ray
"""

''' Matteo Briani, Annie Cuyt, Ferre Knaepkens, Wen-shin Lee,
    VEXPA: Validated EXPonential Analysis through regular sub_sampling,
    Signal Processing 
    Volume 177,
    2020
'''

import numpy as np
from scipy import linalg


import myFunctions as mf
import myDBSCAN as clustering


def shift_decimate(signal, time, u, t0_shift):
    '''
    Decimate signals

    Parameters
    ----------
    signal : input signal.
    time : time array.
    u : decimate factor.
    t0_shift : shift factor.

    Returns
    -------
    signal_sub : decimate signal.
    time_sub : decimate time array.
    '''
    
    index = np.arange(t0_shift, signal.size, u)      
    time_sub = time[index]
    signal_sub = signal[index]
            
    return signal_sub, time_sub
    
    
def solve_ESPRIT(signal, order):
    '''
    Estimation of Signal Parameters via Rotational Invariance 
    Thechniques (ESPRIT).
    
    Roy. R.; Kailath, T. (1989). "Esprit- Estimation of Signal Parameters Via 
    Rotational Invariance Techniques". IEEE Transactions on Acoustics, Speech, 
    and Signal Processing. 37 (7): 984–995.
    
    Parameters
    ----------
    signal : input signal.
    order : selected order. 

    Returns
    -------
    zeta : Eigenvalues corresponding Signal parameters.
    '''
    N = len(signal)
    H = linalg.hankel(signal[:int((N+1)/2)], signal[int((N+1)/2):])
   
    U, _, _ = linalg.svd(H)
    Ul = U[:-1, :order]; Uf = U[1:, :order]
    Phi = linalg.pinv(Ul) @ Uf
    zeta, _ = linalg.eig(Phi)
    
    return zeta

def solve_Vandermonde(signal, zeta):
    '''
    Solve Vandermonde system to obtain the amplitudes of the sum
    of exponentials.

    Parameters
    ----------
    signal : input signal
    zeta : array of eigenvalues

    Returns
    -------
    c : estimated amplitudes
    '''
    N = len(signal)
    #V = np.fliplr(np.vander(zeta, N)).T
    V = np.vander(zeta, N, increasing=True).T
    c = np.dot(linalg.pinv(V), signal)
    return c

def de_aliasing(b1, u, b2, s, threshold = 0.1):
    '''
    Recover from aliasing.
    '''
    b1 = np.power(b1, 1/u)
    b2 = np.power(b2, 1/s)
    
    damp1 = abs(b1)
    b1 = b1/damp1
    b2 = b2/abs(b2)
    nterms = len(b1)
    bu = np.zeros((nterms, u), dtype = complex)
    bs = np.zeros((nterms, s), dtype = complex)
    bu[:,0] = b1; bs[:,0] = b2
    exp_ang1 = np.exp(2*np.pi*1j/u)
    for ii in np.arange(1,u):
        bu[:,ii] = bu[:,ii-1]*exp_ang1
    exp_ang2 = np.exp(2*np.pi*1j/s)
    for ii in np.arange(1,s):
        bs[:,ii] = bs[:,ii-1]*exp_ang2
        
    # Intersect
    distance_matrix = np.zeros((s,u), dtype = float)
    final_b = []
    final_indx = 0
    for ii in range(nterms):
        for jj in range(s):
            distance_matrix[jj,:] = abs(bu[ii,:]-bs[ii,jj]*np.ones(u))
        indx = np.argmin(distance_matrix.flatten('F'))
        value = distance_matrix.flatten('F')[indx]
        col = int(np.ceil(indx/s))
        row = np.remainder(indx,s)
        if row == 0:
            row = s
        if value <= threshold:
            final_b.append(bu[ii,col]*damp1[ii])
            final_indx+=1
            
    b = np.array(final_b, dtype = complex)
    
    b = b[~np.isnan(b)]
    
    return b, b.size
            
def myVEXPA(signal, time, decimation_factor, shift, M, DBSCAN_eps, DBSCAN_minpts, order_init):
    '''
    VEXPA: Validated EXPonential Analysis through regular sub_sampling
    Briani, Cuyt, Knaepkens, Lee. Signal Processing Elsevier, 2020
    
    Parameters
    ----------
    signal : Sum of complex exponential with uknown order.
        
    time : Time array.
        
    decimation_factor : sub-sampling rate.
    
    shift : shift for de-aliasing.
        
    M : number of samples for retrieving s_lambda_i.
    
    DBSCAN_eps : DBSCAN epsilon.
        DBSCAN_eps = [eps_u, eps_s] two epsilon for clsuter lambda_u and lambda_s
        
    DBSCAN_minpts : DBSCAN minpts.
        DBSCAN_minpts = [minpts_u, minpts_s] two minpts for clister lambda_u and lambda_s
    order_init : Initial order.

    Returns
    -------
    b : frequency estimation.
    order_fin: Estimated order.

    '''            
    u = decimation_factor # u in paper.
    s = shift # s in paper
    
    eps_u = DBSCAN_eps[0]; eps_s = DBSCAN_eps[1]
    minpts_u = DBSCAN_minpts[0]; minpts_s = DBSCAN_minpts[1]
   
    alpha_m = np.zeros((M, order_init), dtype=complex)

    for kk in range(u):
        sub_m = []
        time_m = []
        for mm in range(M):
       
            t0_shift = kk + mm*s
            signal_aux, time_aux = shift_decimate(signal, time, u, t0_shift)
        
            Fs_new = 1/(time_aux[1]-time_aux[0])
            time_m.append(time_aux)
            sub_m.append(signal_aux)
    
    
        lambda_u = solve_ESPRIT(sub_m[0], order_init)
    
        if np.count_nonzero(alpha_m) == 0:
            nb = lambda_u.size
            order_init = nb
            alpha_m = np.zeros((M,nb), dtype=complex)
            uL = np.zeros((u,nb), dtype=complex)
            sL = np.zeros((u,nb), dtype=complex)
        
        
        for mm in range(M):
            alpha_m[mm,:] = solve_Vandermonde(sub_m[mm], lambda_u)
        
        if M>2:
            lambda_s = np.zeros(nb, dtype=complex)
            for ii in range(nb):
                lambda_s[ii] = solve_ESPRIT(alpha_m[:,ii], 1)
        elif M==2:
            lambda_s = alpha_m[1,:]/alpha_m[0,:]
        
        uL[kk,:] = lambda_u
        sL[kk,:] = lambda_s
    
    # Perform Cluster analysis on uL

    
    _, uL_cl = clustering.find_cluster(uL, epsilon = eps_u, minpts = minpts_u)
    ncluster = int(max(uL_cl))
   
    
    uL_id = np.arange(len(uL_cl))
    sL = sL.flatten('F').reshape(sL.size,1)
    
    bu = np.zeros(ncluster, dtype = complex)
    bs = np.zeros(ncluster, dtype = complex)

    b_uLc = []

    for cc in range(ncluster):
        uL_cl_EQ_c = (uL_cl==cc+1)
        sLc = sL[uL_cl_EQ_c,:]
        ids = uL_id[uL_cl_EQ_c]
        _, cl = clustering.find_cluster(sLc, epsilon = eps_s, minpts = minpts_s, marker = 'x')
        ncluster2 = int(max(cl))
       
        if ncluster2 == 0:
            # The sL points corresponding to cluster cc do not cluster, hence they 
            # are noise points that accidentally ended up in a cluster, usually the
            # number of points is smaller than u.
            bu[cc] = np.nan + 1j*np.nan
            bs[cc] = np.nan + 1j*np.nan
        else:
            if ncluster2 == 1:
                # There is one cluster, keep only the points in the cluster
                subset = (cl==1)
                sLc = sLc[subset,:]
                ids = ids[subset] # these indices made it into cluster, and thus contribute to the true b-term
                uL_id[ids] = 0 # In the end the nonnzero entries in uL_id all correspond to noise terms
            else:
                # There is more than one cluster: ???
                # find the cluster with the most elements
                
                unique, counts = np.unique(cl, return_counts=True)
                subset = (cl==unique[np.argmax(counts)])
                sLc = sLc[subset,:]
                ids = ids[subset]
                uL_id[ids] = 0
            
            subset = (uL_cl == cc+1)
            uLc = uL.flatten('F')[subset]
            b_uLc.append(uLc)
            bu[cc] = np.mean(uLc)
            bs[cc] = np.mean(sLc)
        
        #uL_noise = uL[uL_id != -1]
    print(len(bu))
    isanumber = ~np.isnan(abs(bu))
    bu = bu[isanumber]
    #b_uLc = np.array(b_uLc)[isanumber]
    bs = bs[isanumber]
    b, order_fin = de_aliasing(bu, u, bs, s)
    #order_fin = ncluster
    return b, order_fin


if __name__ == main:

    # Example taken from 
    # Briani, Cuyt, Knaepkens, Lee. "Validated EXPonential Analysis through regular 
    #                               sub_sampling", Signal Processing Elsevier, 2020
    import matplotlib.pyplot as plt

    r = 12 # Order
    phi = np.zeros(r, dtype = complex)
    alpha = np.zeros(r, dtype = complex)
    i2pi = 2*np.pi*1j

    phi[0] = 0 - i2pi * 5.93
    phi[1] = 0 - i2pi * 4.05
    phi[2] = 0 - i2pi * 3.10
    phi[3] = 0 - i2pi * 1.82
    phi[4] = 0 - i2pi * 1.31
    phi[5] = 0 + i2pi * 1.90
    phi[6] = 0 + i2pi * 2.97
    phi[7] = 0 + i2pi * 6.05
    phi[8] = 0 + i2pi * 6.67
    phi[9] = 0 + i2pi * 38.0
    phi[10] = 0 + i2pi * 43.0
    phi[11] = 0 + i2pi * 24.0

    alpha[0] = 1 * np.exp(0 * 1j)
    alpha[1] = 2 * np.exp(np.pi * 1j)
    alpha[2] = 2 * np.exp(np.pi/4 * 1j)
    alpha[3] = 2 * np.exp(np.pi/8 * 1j)
    alpha[4] = 2 * np.exp(3*np.pi/4 * 1j)
    alpha[5] = 1 * np.exp(np.pi/10 * 1j)
    alpha[6] = 3 * np.exp(-np.pi * 1j)
    alpha[7] = 1.5 * np.exp(-7*np.pi/8 * 1j)
    alpha[8] = 2 * np.exp(0 * 1j)
    alpha[9] = 3 * np.exp(-78*np.pi/100 * 1j)
    alpha[10] = 1 * np.exp(0 * 1j)
    alpha[11] = 1 * np.exp(np.pi/5 * 1j)

    Fs = 100
    N = 300
    b = np.exp(phi/Fs)
    c = alpha
    nb = len(b)
    terms = np.zeros((N, nb), dtype = complex)
    terms[0, :] = c*b**(0*Fs)
    for ii in np.arange(1,N):
        terms[ii,:] = terms[ii-1,:]*b

    samples = np.sum(terms, axis = 1)
    time = np.arange(0, N, 1/Fs)[:N]

    u = 7
    s = 6
    M = 8
    order_initial = 15

    DBSCAN_eps = [0.1, 0.1]
    DBSCAN_minpts = [6, 5]

    y, _ = mf.add_noise(samples, 25)

    b_hat, r_hat = myVEXPA(y, time, u, s, M, DBSCAN_eps, DBSCAN_minpts, order_initial)

    plt.figure()
    plt.plot(b1.real, b1.imag, 'kX')
    plt.plot(b.real, b.imag, 'ro', mfc = 'none')
    plt.grid(True)
    plt.show()

'''       
N = 128; K = 2*N+1
t = np.arange(-0.5,0.5,1/K); Ts = 1/K; Fs = 1/Ts
snrdB = np.linspace(-10,10,15)
Niter = 200

xi_1 = np.array([-0.274-7.68*1j, -0.150+39.68*1j, 0.133+40.96*1j, 
                     -0.221+99.84*1j], dtype = complex)*2*np.pi
a_1 = np.array([0.4*np.exp(-1j*0.93), 1.2*np.exp(-1j*1.55), 
                    1.0*np.exp(-0.83*1j), 0.9*np.exp(1j*0.07)], dtype = complex)

xi_true = np.outer(np.log(np.sort(np.exp(xi_1/K)))*K/(2*np.pi),np.ones(Niter))

r = len(a_1)


x1 = np.matmul(a_1,np.exp(np.outer(xi_1,t)))

K = 257  # Numbers of samples
N = int((K-1)/2)
t = np.arange(K)
r = 10
amp = np.exp(1j*np.random.uniform(0, 2*np.pi,r))
omega = 2*np.pi*np.random.uniform(0,1,r)
x = np.matmul(amp,np.exp(np.outer(1j*omega,t)))
      
u = 7 
s = 6
M = 8
order_init =13

DBSCAN_eps = [0.1, 0.1]
DBSCAN_minpts = [3,3]

y, _ = mf.add_noise(x, 25)


b, r_est = myVEXPA(y, t, u, s, M, DBSCAN_eps, DBSCAN_minpts, order_init)      
'''     
            
    
    

        
        