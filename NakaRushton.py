#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:14:51 2021

@author: joshuafoster
"""

def NakaRushton(c,b,Gr,Gc,n): 
    """
    Naka-Rushton function

    Parameters
    ----------
    c (float for single value, numpy array for multiple values): contrast value(s)
    b (float): baseline
    Gr (float): response gain
    Gc (float): contrast gain
    n (float): slope

    Returns
    -------
    r (same as c param): response(s) for given contrast levels
    
    """
 
    r = Gr*((c ** n)/((c ** n)+(Gc ** n)))+b;
    return r


def computeRmax(b,Gr,Gc,n):
    """
    Calculate Rmax (i.e. value at 100% contrast minus baseline)

    Parameters
    ----------
    b (float): baseline
    Gr (float): response gain
    Gc (float): contrast gain
    n (float): slope

    Returns
    -------
    Rmax (float)     
    
    """
    
    Rmax = NakaRushton(100,b,Gr,Gc,n)-b
    return Rmax   


def computeC50(b,Gr,Gc,n):
    """
    Calculate C50 (contrast where function is halfway between baseline and value at 100% contrast)
    
    Parameters
    ----------
    b (float): baseline
    Gr (float): response gain
    Gc (float): contrast gain
    n (float): slope

    Returns
    -------
    C50 (float)
       
    """

    import numpy as np
    
    maxResp = NakaRushton(100,b,Gr,Gc,n)
    semiSatResp = (b + maxResp)/2
    c = np.arange(0.01,100,0.001)
    pred = NakaRushton(c,b,Gr,Gc,n)
    idx = np.argmin(np.abs(pred-semiSatResp))
    C50 = c[idx]
    return C50


def fitNakaRushton(contrast,resp,init_params=[0,1,50,3],lower_bounds=[-10,0,0,0.1],upper_bounds=[10,10,100,10]):
    """
    fit Naka-Rushton function to data

    Parameters
    ----------
    contrast (numpy array): vector of contrast values
    resp (numpy array): vector of response values
    
    OPTIONAL ARGS:
    init_params (list): initial guessing for params [b,Gr,Gc,n]
    lower_bounds (list): parameter lower bounds [b,Gr,Gc,n]
    upper_bounds (list): parameter upper bounds [b,Gr,Gc,n]

    Returns
    -------
    b, Gr, Gc, n, Rmax, C50 (floats) 

    """
    
    from scipy.optimize import curve_fit
       
    popt, pcov = curve_fit(NakaRushton, contrast, resp, p0 = init_params, bounds = (lower_bounds,upper_bounds))
    
    b = popt[0]
    Gr = popt[1]
    Gc = popt[2]
    n = popt[3]
    Rmax = computeRmax(b,Gr,Gc,n)
    C50 = computeC50(b,Gr,Gc,n)
    
    return b, Gr, Gc, n, Rmax, C50