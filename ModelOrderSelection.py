#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 12:04:33 

Order Estimation Methods

@author: ray
"""

import numpy as np
from scipy import linalg, integrate

def ESTER(H):
    '''
    Estimation Error (ESTER) method
    
    Badeau, R., et. al. "Selecting the modeling order for the ESPRIT high 
    resolution method: an alternative approach". Proc. of IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP), 2004,.

    Parameters
    ----------
    H : Hankel matrix built from signal with unknown model order

    Returns
    -------
    J : ESTER cost function
    order_est: Estimated order argmin(J).
    '''
    
    U, _, _, = linalg.svd(H)
    p = U.shape[0]
    L = int(p/2.0)
    J = np.zeros(L, dtype = float)
    
    for ii in range(L):
        Uf = U[1:, :ii+1]; Ul = U[:-1, :ii+1]
        E = Uf - Ul @ linalg.pinv(Ul) @ Uf
        J[ii] = linalg.norm(E, ord = 2)
        
    order_est = np.argmin(J) + 1
    return order_est, J

##############################################################################

def SAMOS(H):
    '''
    Subspace-based Automatic Model Order Selection (SAMOS) method.
    
    Papy, J., et. al. "Invariance-Based Order-Selection Technique for 
    Exponential Data Modelling". IEEE Signal Processing Letters, 2007.
    Parameters
    ----------
    H : Hankel matrix built from signal with unknown model order

    Returns
    -------
    order_est : Estimated order argmin(J).
    J : SAMOS cost function

    '''
    
    U, _, _ = linalg.svd(H)
    p = U.shape[0]
    L = int(p/2.0)
    J = np.zeros(L, dtype = float)
    
    for ii in range(L):
        Uf = U[1:, :ii+1]; Ul = U[:-1, :ii+1]
        E_aug = np.concatenate((Uf, Ul), axis = -1)
        gamma = linalg.svd(E_aug, compute_uv = False)
        J[ii] = np.sum(gamma[ii+1:])/(ii+1)
        
    order_est = np.argmin(J) + 1
    return order_est, J
    
#############################################################################

def MUSIC(y):
    '''
    MUSIC method for model order selection

    Christensen, M., et. al. "Sinusoidal order estimation using angles
    between subspaces". Eurasip Journal Adv. Signal Processing, 2009.

    Parameters
    ----------
    y : noisy sum of sum of exponentials functions with unknown order

    Returns
    -------
    order_est : Estimated order.

    '''

    N = len(y); M = int((N-1)/2.0)
    L = int(M/4.0)
    R = np.zeros((M, M), dtype = complex)
    J = np.zeros(L, dtype = float)
    
    for ii in range(N-M):
        R = R + np.outer(y[ii:ii+M], np.conj(y[ii:ii+M]))/(N-M)
       
    _, U = linalg.eig(R)
    for ii in range(L):
        E_aux = U[:, ii:]
        E = E_aux @ E_aux.conj().T
        a = np.zeros(2*M-1, dtype=complex)
        for jj in np.arange(-M+1, M):
            a[jj+M-1] = np.sum(np.diag(E, jj))
        ra = np.roots(a[::-1])
        rb = ra[abs(ra)<1]
        idx = np.argsort(abs(abs(rb)-1))
        z = rb[idx[:ii+1]]
        K = int(np.min([ii+1,M-(ii+1)]))
        A = np.fliplr(np.vander(z,M)).T
        J[ii] = linalg.norm(A.conj().T @ E, ord = 'fro')**2/(M*K)
        
    order_est = np.argmin(J) + 1
    return order_est    
        
#############################################################################

def ModelOrderSelection_Constrain(H, beta = 0.1, eta = None, func = 'ESTER'):
    '''
    
    Constraint Optimization Problem for Model Order Estimation.

    [1] Albert, R., Galarza. C, "A Constraint Optimization Problem for Model Order
    Estimation". Signal Processing, 2024.


    Parameters
    ----------
    H : Hankel matrix.
    beta : probability bound such that P(|H_w|>tau)>beta
        The default is 0.1.
    eta : noise variance
        The default is None.
    func : Cost function
        The default is 'ESTER'.

    Returns
    -------
    idx_order : Estimated order.

    '''
    
    m, n = H.shape
    K = m+n-1; N = int(n/2)
    
    singValues = linalg.svd(H, compute_uv = False)
    
    if eta == None: # Unknown noise variance see [1]
        coeff = np.array([0.50080263, -0.19149437]) # See [1]
        delta = np.exp(coeff[1])*np.sqrt(n)
        eta = np.median(singValues)/delta
    
    
    if func == 'ESTER':
        _, J = ESTER(H) 
    else:
        _, J = SAMOS(H)
        
    
    alpha = np.sqrt(-eta**2*np.log(1.0-np.power(1-beta,1/K)))
    tau = alpha*np.sqrt(K)
    
    aux = np.zeros(N, dtype = float)
    for nn in range(N):
        aux[nn] = singValues[nn+1]
      
    aux2 = np.nonzero(aux>tau)
    J[aux2] = np.nan
    idx_order = np.nanargmin(J) + 1
    
    return idx_order

#############################################################################

def HardThreshold(H, eta = None, Noise = 'Gaussian' , delta = 0.1):
    '''
    Hard threshold for singular values

    Parameters
    ----------
    H : Hankel matrix
    eta : noise variance
        DESCRIPTION. The default is None.(unknown variance)
    Noise : Type of Noise
        DESCRIPTION. The default is 'Gaussian'.
    delta: probability of a singular values will be grater than some tau

    Returns
    -------
    Thind = Number of relevant singular values (rank of matrices).
    '''
    
    m, n = H.shape
    K = m + n - 1
    if m < n:
        beta = m / n
    else:
        beta = n / m
    
    singValues = linalg.svd(H, compute_uv = False)
    
    if eta == None:
        if Noise == 'Gaussian': # Noisy matrix with Gaussian iid entries.
             ''' See Gavish, Donoho. " The optimal hard Threshold for singular values
             is 4/\sqrt{3}". IEEE Transaction Information Theory, 2014.'''
             th = np.sqrt(2*(beta+1)+8*beta/(beta + 1 + np.sqrt(beta**2 + 14*beta + 1)))/np.sqrt(MedianMarcenkoPastur(beta))*np.median(singValues)
            
        else: # Noisy Hankel matrix from Gaussian vector.
             ''' See Albert, Galarza. "A Constraint Optimization Problem for Model Order
                 Estimation". Signal Processing, 2024.'''
             coeff = np.array([0.50080263, -0.19149437]) 
             gamma = np.exp(coeff[1])*np.sqrt(n)
             eta = np.median(singValues)/gamma
             th = np.sqrt(-eta ** 2 * np.log(1 - (1-delta) ** (1 / K)))*np.sqrt(K)
         
    
    thind = np.argmax([ii for ii, x in enumerate(singValues) if x > th]) 
    
    return thind+1


''' The following functions are taking from Gavish, Donoho. " The optimal hard Threshold for singular values
             is 4/\sqrt{3}". IEEE Transaction Information Theory, 2014.'''
def MarPas(x, UpperBound, LowerBound, beta):
    if (UpperBound - x) * (x - LowerBound) > 0:
        return np.sqrt((UpperBound - x) * (x - LowerBound)) / (beta * x) / (2 * np.pi)
    else:
        return 0


def MedianMarcenkoPastur(beta):
    LowerBound = lobnd = (1 - np.sqrt(beta)) ** 2
    UpperBound = hibnd = (1 + np.sqrt(beta)) ** 2
    change = 1

    while (change & ((hibnd - lobnd) > .001)):
        change = 0
        x = np.linspace(lobnd, hibnd, 10)
        y = np.zeros_like(x)
        for ii in np.arange(len(x)):
            yi, err = integrate.quad(MarPas, a=x[ii], b=UpperBound, 
                                     args=(UpperBound, LowerBound, beta))
            y[ii] = 1 - yi

        if np.any(y < 0.5):
            lobnd = np.max(x[y < 0.5])
            change = 1

        if np.any(y > 0.5):
            hibnd = np.min(x[y > 0.5])
            change = 1

    return (hibnd + lobnd) / 2.

#############################################################################

def gPrior_BIC(y, lk, Ts, delta = 2):
    '''
    Bayesian information theoretic criteria for model order selection.

    Nielsen, J. K., et. al. "Bayesian model comparasion with the g-prior".
    IEEE Transaction on Signal Processing, 2014.

    '''

    N = len(y)
    Z = EstimationFrequencies(y, lk, int(N/4))

    Pz = Z @ linalg.pinv(Z)
    u = N/2
    w = (N-lk-delta)/2
    v = 1
    R = np.matmul(np.matmul(y.conj().T, Pz), y)/(np.dot(y.conj(), y))

    alpha_tau = (1-R)*(v+w-u)
    beta_tau = (u-v)*R + 2*v+w-u 

    tau_hat = np.log((beta_tau+np.sqrt(beta_tau**2-4*alpha_tau*v))/(-2*alpha_tau))
    g_hat = np.exp(tau_hat)
    gamma_g = 1./(g_hat*u*(1-R)/(1+g_hat*(1-R))**2 - g_hat*w/(1+g_hat)**2)

    sigma2_k = np.matmul(np.matmul(y.conj().T, (np.diag(np.ones(N)) - g_hat/(g_hat+1)*Pz)), y)/N
    D = Hessian(y, Z, lk, Ts)
    coeff = np.power(N*np.exp(tau_hat)/(2*N*(1+np.exp(tau_hat))*sigma2_k),2)
    
    detH = linalg.det(-coeff*D)
   
    
    k_hat = -lk/2*np.log(1+g_hat) - N/2*np.log(sigma2_k) + np.log(g_hat) \
        - delta/2*np.log(1+g_hat) + (2*lk+1)/2*np.log(2*np.pi) \
        + 0.5*np.log(gamma_g) - 0.5*np.log(detH)
    
            
    return k_hat.real
                                    

def EstimationFrequencies(y, r, M):
    '''
    Estimation frequencies from covariance matrix
    '''

    N = len(y)
    R = np.zeros((M, M), dtype = complex)

    for ii in range(N-M):
        R = R + np.outer(y[ii:ii+M], np.conj(y[ii:ii+M]))/(N-M+1)

    _, U = linalg.eig(R)
    E = U[:, r:]
    EE = np.matmul(E, E.conj().T)

    a = np.zeros(2*M-1, dtype = complex)
    for ii in np.arange(-M+1,M):
        a[ii+M-1] = np.sum(np.diag(EE, ii))

    ra = np.roots(a[::-1])
    rb = ra[abs(ra)<1]
    idx = np.argsort(abs(abs(rb)-1))
    z = np.sort(rb[idx[:r]])

    Z = np.fliplr(np.vander(z, N)).T

    return Z

def Hessian(y, Z, r, Ts):

    N = len(y)
    mk = np.dot(linalg.pinv(Z), y)
    Pz = np.matmul(Z, linalg.pinv(Z))
    I1 = 2*np.pi*np.arange(N).reshape(N,1)
    IN = np.diag(np.ones(N))

    T = 1j*I1*Z*mk
    D = -2*np.real(np.matmul(np.matmul(T.conj().T, IN-Pz), T))

    return D
