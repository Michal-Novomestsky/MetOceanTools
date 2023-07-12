#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 11:59:04 2017

@author: hrap
"""

def rh2ah(RH,p,T):
    '''conversion relative humidity to absolute humidity (kg Water per m^3 Air) '''
    eee=RH*exp(28.48859-0.0091379024*T-6106.396/T)
    q=eee/(p-0.00060771703*eee)

    return  q