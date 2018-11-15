#!/usr/bin/env python

from scipy.special import erf
#from mpmath import *

def hiprec_erf(x):
    #mp.dps = 1000
    return erf(x)
