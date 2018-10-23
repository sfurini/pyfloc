#!/usr/bin/env python

from scipy.special import erf
from mpmath import *
mp.dps = 1000

def hiprec_erf(x):
    return erf(x)
