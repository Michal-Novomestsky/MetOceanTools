#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:03:24 2017

@author: hrap
"""

def esat(T):
    ''' get saturation pressure (units [Pa]) for a given air temperature (units [K])'''
    from numpy import log10
    
    TK = 273.15
    e1 = 101325.0
    logTTK = log10(T/TK)
    esat =  e1*10**(10.79586*(1-TK/T)-5.02808*logTTK+ 1.50474*1e-4*(1.-10**(-8.29692*(T/TK-1)))+ 0.42873*1e-3*(10**(4.76955*(1-TK/T))-1)-2.2195983) 
    return esat

def rh2mixr(RH,p,T):
    ''' conversion relative humidity (unitless) to mixing ratio [kg/kg]'''
    Mw=18.0160 # molecular weight of water
    Md=28.9660 # molecular weight of dry air

    es = esat(T)
    return Mw/Md*RH*es/(p-RH*es)

def spechum(RH,p,T):
    '''conversion from mixing ratio (units [kg/kg]) to specific humidity (units also [kg/kg])'''
    
    W = rh2mixr(RH,p,T)
    
    return W/(1.+W)

