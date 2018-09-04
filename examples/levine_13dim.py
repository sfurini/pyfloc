#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
import sys
sys.path.append('../pyfloc')

import pyfloc

B = pyfloc.PyFloc(prefix = 'levine_13dim', verbose = 2)
B.read_input('./levine_13dim.txt')

exit()
