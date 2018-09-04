#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
import sys
sys.path.append('../pyfloc')
from matplotlib.backends.backend_pdf import PdfPages

import pyfloc
import data

pdf = PdfPages('./test_clustering.pdf')
B = pyfloc.PyFloc(prefix = 'nilsson_rare', verbose = 2)
B.read_input('./nilsson_rare.txt')
#remove_outliers = CD110,CD34,CD4,CD19,CD10,CD11b,CD45,CD45RA,CD49fpur,CD38,CD123,CD90bio,CD3
B.experiments.show_scatter(['CD38', 'CD90bio'], pdf = pdf, stride = 1)
B.experiments.show_scatter(['CD110', 'CD34'], pdf = pdf, stride = 1)

#E = data.Experiment('./data/flowc/nilsson_rare.fcs')
#C = data.Collection()
#C.add_experiment(E, condition = 'test1', labels = E.get_data_features('label'))
#C.show_scatter(['CD38', 'CD90bio'], pdf = pdf, stride = 1)

pdf.close()

exit()
