#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('../pyfloc')
import pyfloc


B = pyfloc.PyFloc(verbose = 2, prefix = 'levine_13dim')
B.read_fcs(file_name = './data/flowc/levine_13dim.fcs', mode = 40000)
list_features= ['CD34','CD123','CD19','CD33','CD20','CD38','CD11b','CD4','CD8','CD90','CD45RA','CD45','CD3']
B.clean_samples(features = ['label',], mode = 'nan')
#B.normalize(features = list_features, mode = 'arcsinh', factor = 5.0)
B.normalize(features = list_features, mode = 'logicle')
B.experiments.remove_outliers(list_features, 6.0)
#B.write(file_name = 'levine_13.pk')

#pdf = PdfPages('levine_13.pdf')
#with open('levine_13.pk','rb') as fin:
#    B = pickle.load(fin)
#    B.counter = 0

B.fit_cluster(list_features, ns_clusters = np.arange(2,50,1), percents = [0.01, 0.1], mode = 'DP')
B.predict_cluster()
B.counter += 1
#B.fit_cluster(list_features, ns_clusters = np.arange(2,50,1), percents = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0], mode = 'DP')
#B.fit_cluster(list_features, ns_clusters = np.arange(2,50,1), percents = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0], mode = 'DP')
#B.fit_cluster(list_features, ns_clusters = np.arange(2,50,1), percents = [1e-7,1e-6,1e-5,1e-4,1e-3,0.01,0.1,1.0,10.0], mode = 'DP')
#for p in [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]:
#    for p2 in [1,2,3,4,5,6,7,8,9]:
#        B.fit_cluster(list_features, ns_clusters = np.arange(2,50,1), percents = p*p2, mode = 'DP')
#        B.predict_cluster()
#        #B.experiments.show_distributions(list_features, pdf = pdf)
#        B.counter += 1
#        print(B.cluster)

#pdf.close()
exit()
