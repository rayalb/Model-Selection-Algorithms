#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:17:12 2022

@author: ray
"""

'''
Auxiliary functions

'''

import numpy as np
from scipy import interpolate, signal

def nextpow2(x):
    '''
    Compute the integer exponent of the next higher power of 2.
    
    Parameters
    ----------
    x = Number whose next higest power of 2 needs to be computed.

    Returns
    -------
    Exponent of the next higher power of 2

    '''
    
    return 2**int(np.ceil(np.log2(x)))

def fft_comp(inSignal, fs, z_padfact = 4):
    '''
    Compute the Fast Fourier Transform
    
    Parameters
    ----------
    inSignal : Input Signal.
    fs: Sample Frequency.
    z_padfact : multiplier the lenght of the signal to 
                obtain lenght of FFT. Default is 4.

    Returns
    -------
    fftSignal : FFT of inSignal of lenght z_padfact*inSignal.shape[0].
    freqs : arrays of frequencies.
    '''
    nfft = nextpow2(inSignal.shape[0]*z_padfact)
    fftSignal = np.fft.fftshift(np.fft.fft(inSignal, n = nfft, axis = 0), axes=(0,))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs))
    return fftSignal, freqs

def filt_signal(input_signal, Ntaps, fcutoff, Fs, window = 'hamming'):
    '''
    Filter input signal with a FIR Filter.

    Parameters
    ----------
    input_signal : Input Signal.
    Ntaps : Number of filter coefficents.
    fcutoff : Cut off Frequencies. See signal.firwin().
    Fs : Sample Frequency.
    window : Window used to build the Filter. Default 'hamming'.

    Return
    ------
    filter_x : Filtered Signal.
    '''
    taps = signal.firwin(Ntaps, fcutoff, window = window)
    filter_x = signal.lfilter(taps, 1.0, 
                              np.append(input_signal, 
                                        np.zeros(int(Ntaps/2))))[int(Ntaps/2)-1:-1]
    return filter_x

#############################################################################

def add_noise(x, snrdB):
    '''
    Add noise for a given SNR in dB
    Parameters
    ----------
    x : array
        noiseless signal.
    snrdB : float
        SNR in dB.

    Returns
    -------
    y : array
        noisy signal. y = x + e
    e: array
        noise vector.

    '''
    L = len(x)
    SNR = np.power(10, snrdB/10.0)
    E_x = sum(abs(x)**2)/L
    E_n = E_x/SNR
    
    if np.isrealobj(x):
        sig_noise = np.sqrt(E_n)
        e = sig_noise*np.random.randn(L)
    else:
        sig_noise = np.sqrt(E_n/2)
        e = sig_noise*(np.random.randn(L)+1j*np.random.randn(L))
        
    y = x + e
    return y, e

#############################################################################  

def add_color_noise(x, a, snrdB):
    '''
    Add colored noise for a given SNR in dB. The colored noise is generated 
    by an AR1 model.

    Parameters
    ----------
    x : noiseless signal.
    a : Parameter of the AR1 filter.
    snrdB : snr in dB.

    Returns
    -------
    y: Noisy signal y = x+w
    w: Noise signal

    '''
    L = len(x)
    SNR = np.power(10, snrdB/10)
    
    E_x = sum(abs(x)**2)/L
    E_n = E_x/SNR
    
    if np.isrealobj(x):
        sig_noise = np.sqrt(E_n)
        e = sig_noise*np.random.randn(L)
    else:
        sig_noise = np.sqrt(E_n/2)
        e = sig_noise*(np.random.randn(L)+1j*np.random.randn(L))
    
    w = signal.lfilter([1], [1, a], e)
    y = x + w
    return y, w

#############################################################################

def HistInterp(data, X):
    '''
    Histogram interpolation
    
    Parameters
    ----------
    data : data to generate histogram.
    X : bins.

    Returns
    -------
    N : interpolated histogram 
    t : x-axis

    '''
    N,  edges = np.histogram(data, X, density = True)
    edges_new = edges[1:] - (edges[1]-edges[0])/2
    t = np.linspace(edges_new[0], edges_new[-1], 1000)
    f = interpolate.PchipInterpolator(edges_new, N)
    N = f(t)
    
    return N, t    
    