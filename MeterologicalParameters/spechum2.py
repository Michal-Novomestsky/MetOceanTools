#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:03:24 2017

@author: hrap
"""
from numpy import log10,exp

Mw=18.0160 # molecular weight of water
Md=28.9660 # molecular weight of dry air

def spechum2(RH,p,T):
    '''conversion from mixing ratio (units [kg/kg]) to specific humidity (units also [kg/kg])'''
    ''' below is from https://wiki.usask.ca/pages/viewpage.action?pageId=380797075 '''
    a=((0.7859+0.03477*T)/(1+0.00412*T)+2)
    ea=RH/100*(10**a) # site gives RH in percent but is unitiless
    qa = (Mw/Md*ea)/(p-0.738*a) # Mw/Md~0.622
    
    return qa
