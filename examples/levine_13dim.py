#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('../pyfloc')
import pyfloc
from copy import deepcopy
from scipy.signal import argrelextrema


pdf = PdfPages('levine_13dim_bin_all_possible_strategies.pdf')
B = pyfloc.PyFloc(verbose = 2, prefix = 'levine_13dim')
B.read_fcs(file_name = './data/flowc/levine_13dim.fcs', mode = 10000)
list_features= ['CD34','CD123','CD19','CD33','CD20','CD38','CD11b','CD4','CD8','CD90','CD45RA','CD45','CD3']
# CD38 = non binarizzabile (k = 1.0, 0.25)
# CD33 = binarizzabile con k = 0.25
#list_features= ['CD38', 'CD33']
B.clean_samples(features = ['label',], mode = 'nan')
B.experiments.remove_outliers(list_features, 6.0)

B.normalize(features = list_features, mode = 'min')
#B.normalize(features = list_features, mode = 'min_max')
#B.normalize(features = list_features, mode = 'mean_std')
#B.normalize(features = list_features, mode = 'arcsinh', factor = 5, bias = 0.0)
#B.normalize(features = list_features, mode = 'logicle')
#B.experiments.show_scatter(['CD4','CD8'], mode = 'density', pdf = pdf)
#B.experiments.show('CD4','CD8', pdf = pdf)

#B.write(file_name = 'levine_13.pk')
#with open('levine_13.pk','rb') as fin:
#    B = pickle.load(fin)
#    B.counter = 0

#B.fit_cluster(list_features, mode = 'Kmeans', ns_clusters = 40)
#B.predict_cluster()

radius = 0.1375 # radius of the kernel
list_features_binary = []
for feature in list_features:
    B.experiments.show_histogram(pdf, list_features = [feature,])
    energy = B.fit_cluster([feature,], mode = 'DP', ns_clusters = [2,3,4,5], radii = radius, metric = 'logarithmic')
    B.predict_cluster()
    #density_peaks_norm = np.sort(B.cluster.clusters_analogic.flatten())
    #density_peaks = B.experiments.back_transform(feature, density_peaks_norm)
    #if (energy[0] > 1.0) and (density_peaks[1]/np.abs(density_peaks[0]) > 10.0):
    #    data_norm = B.experiments.get_data_norm_features([feature])
    #    h, e = np.histogram(data_norm, bins = np.linspace(density_peaks_norm[0], density_peaks_norm[1],100))
    #    b = 0.5*(e[:-1]+e[1:])
    #    i_min = np.argmin(h)
    #    f = plt.figure()
    #    ax1 = f.add_subplot(111)
    #    ax1.plot(b,h,'-b')
    #    ax1.plot(b[i_min],h[i_min],'*r')
    #    plt.title(feature)
    #    pdf.savefig()
    #    plt.close()
    #    list_features_binary.append(feature)
    #    threshold = B.experiments.back_transform(feature, b[i_min])
    #    print('Feature: ',feature,' density peaks: ',density_peaks,' density_peaks_norm = ',density_peaks_norm,' energy = ',energy,' threshold = ',threshold, ' raw = ',b[i_min])
    #    #B.normalize(features = [feature,], mode = 'binary', threshold = threshold)
    #    #B.experiments.stretch([feature,], thrs = {'min':np.min(data_norm), 'peak_low':density_peaks_norm[0], 'threshold':b[i_min], 'peak_high':density_peaks_norm[1], 'max':np.max(data_norm)})
    #    B.experiments.show_histogram(pdf, list_features = [feature,])
    #else:
    #    print('Feature: ',feature,' density peaks: ',density_peaks,' density_peaks_norm = ',density_peaks_norm,' energy = ',energy)
    B.counter += 1
print(B)

#print('List features: ',list_features_binary)
#B.fit_cluster(list_features_binary, mode = 'Unique')
#B.predict_cluster()
#ns_clusters = np.arange(2,50,1)
#energies = B.fit_cluster(list_features_binary, ns_clusters = ns_clusters, radii = radius, mode = 'DP')
#B.predict_cluster()
#energies = energies.flatten()


pdf.close()
exit()
