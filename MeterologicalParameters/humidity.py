#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:41:25 2017

@author: hrap
"""
from math import log10,exp

""" Description:
    set of functions for conversions between different humidity quantities
    Required input: T, p, RH
"""
# constants describing air and water properties
cpd  = 1005.7 # isobaric specific heat of dry air at constant pressure [J/(kg K)]
cpv = 1864. # isobaric specific heat of water vapour at constant pressure and 300K [J/(kg K)]
rhod  = 1.2 # specific mass of dry air [kg/m**3] for standard atmosphere
p0 = 101325.0 # reference pressure (Pa)
Rd = 287. # specific gas constant for dry air [J kg^-1 K^-1]
Rv = 462. # specific gas constant for water vapour [J kg^-1 K^-1]
#
Mw=18.0160 # molecular weight of water
Md=28.9660 # molecular weight of dry air
R =  8.31432E3 # gas constant
Rd = R/Md # specific gas constant for dry air
Rv = R/Mw # specific gas constant for vapour
Lv = 2.5e6 # heat release for condensation of water vapour [J kg-1]
eps = Mw/Md

g = 9.81 # gravitational accelleration

def rhov_modified(T,p,sh=0):
    '''Same as rhov but instead accepts T in degC and p in mBar'''
    T += 273.15
    p *= 100
    return rhov(T,p,sh)

def hum2ea_modified(p, spechum):
    '''Converts pressure [mBar] and specific humidity [kg/kg] to vapour pressure [Pa]'''
    p *= 100
    return mixr2ea_modified(p, sh2mixr(spechum))

def mixr2ea_modified(p, mixr):
    '''Converts dry air pressure [Pa] and mixing ratio [kg/kg] to vapour pressure [Pa] (derived by taking inverse of ea2mixr)'''
    return 500*mixr*p/(311 + 500*mixr)

def rh2sh_modified(RH,p,T):
    '''Converts relative humidity [%] to specific humidity [kg/kg] using air pressure [mBar] and temperature [degC]'''
    T += 273.15
    p *= 100
    return mixr2sh(rh2mixr(RH, p, T))

def rhov(T,p,sh=0.):
    '''purpose: calculate the density from pressure, temperature and specific humidity'''
    R = Rd*(1.-sh) + Rv*sh
    return p/R/T

def esat(T):
    ''' get sateration pressure (units [Pa]) for a given air temperature (units [K])'''
    from numpy import log10
    TK = 273.15
    e1 = 101325.0
    logTTK = log10(T/TK)
    esat =  e1*10**(10.79586*(1-TK/T)-5.02808*logTTK+ 1.50474*1e-4*(1.-10**(-8.29692*(T/TK-1)))+ 0.42873*1e-3*(10**(4.76955*(1-TK/T))-1)-2.2195983) 
    return esat
def esat2(T):
    ''' a simpler form for the saturation pressure (units [Pa]) for a given air temperature (units [K]), based on clausius-claperyon'''
    return 611.*exp(-Lv/Rv*(1./T - 1./273.16))

def rh2mixr(RH,p,T):
    '''purpose: conversion relative humidity (unitless) to mixing ratio [kg/kg]'''
    es = esat(T)
    return Mw/Md*RH*es/(p-RH*es)

def mixr2rh(mixr,p,T):
    '''purpose: conversion mixing ratio to relative humidity [kg/kg] (not tested)'''
    return mixr * p/((mixr+Mw/Md)*esat(T))

def mixr2sh(W):
    '''conversion from mixing ratio (units [kg/kg]) to specific humidity (units also [kg/kg])
    '''
    return W/(1.+W)
def sh2mixr(qv):
    '''conversion from specific humidity (units [kg/kg]) to mixing ratio (units also [kg/kg])
    '''
    return qv/(1.-qv)
def ea2mixr(P,ea):
    '''conversion from dry air and vapour pressure (units [Pa]) to mixing ratio (units [kg/kg]) 
    '''
    return 0.622*ea/(P-ea)
def mixr2ea(P,mixr):
    '''conversion from dry air pressure (units [Pa]) and mixing ratio (units [kg/kg] to vapour pressure (units [Pa])
    '''
    return P*mixr/(0.622+ea)

def ah2mixr (rhov,p,T):
    '''conversion from absolute humidity (units [kg/m**3]) to mixing ratio (units also [kg/kg])
       not tested
    '''
    return (Rd * T)/(p/rhov-Rv*T)

# not tested
def wvap2sh(e,p):
    '''conversion from water vapour pressure (units [Pa]) to specific humidity (units [kg/kg]) 
    '''
    return eps*e/(p-(1.-eps)*e)

def rh2ah(RH,p,T):
    '''conversion relative humidity to absolute humidity (kg Water per m^3 Air)'''
    mixr=rh2mixr(RH, p,T)
    sh=mixr2sh(mixr)
    return  sh*rhov(T,p,sh) 
