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

def check_strategy(strategy, list_features_binary):
    check=1
    for strategia in list(strategy.keys()):
        if strategia not in list_features_binary:
            check = 0
    return check

pdf = PdfPages('db_k20.pdf')
B = pyfloc.PyFloc(verbose = 2, prefix = 'db_k20')
B.read_fcs(file_name = './data/flowc/levine_13dim.fcs', mode = 'all')
list_features= ['CD34','CD123','CD19','CD33','CD20','CD38','CD11b','CD4','CD8','CD90','CD45RA','CD45','CD3']
B.clean_samples(features = ['label',], mode = 'nan')
#B.experiments.remove_outliers(list_features, 6.0)

#B.normalize(features = list_features, mode = 'arcsinh', factor = 5, bias = 0.0)
B.normalize(features = list_features, mode = 'logicle')
#B.experiments.show_scatter(['CD4','CD8'], mode = 'density', pdf = pdf)
#B.experiments.show('CD4','CD8', pdf = pdf)

#B.write(file_name = 'levine_13.pk')
#with open('levine_13.pk','rb') as fin:
#    B = pickle.load(fin)
#    B.counter = 0

#B.fit_cluster(list_features, mode = 'Kmeans', ns_clusters = 40)
#B.predict_cluster()
radius = 0.20

list_features_binary = []
for feature in list_features:
    B.experiments.show_histogram(pdf, list_features = [feature,])
    energy = B.fit_cluster([feature,], ns_clusters = [2,], radii = radius, mode = 'DP')
    B.predict_cluster()
    density_peaks_norm = np.sort(B.cluster.clusters_analogic.flatten())
    density_peaks = B.experiments.back_transform(feature, density_peaks_norm)
    if (energy[0] > 1.0) and (density_peaks[1]/np.abs(density_peaks[0]) > 10.0): #prima era 10.0, provare con 3 e radius 0.25
        data_norm = B.experiments.get_data_norm_features([feature])
        h, e = np.histogram(data_norm, bins = np.linspace(density_peaks_norm[0], density_peaks_norm[1],100))
        b = 0.5*(e[:-1]+e[1:])
        i_min = np.argmin(h)
        f = plt.figure()
        ax1 = f.add_subplot(111)
        ax1.plot(b,h,'-b')
        ax1.plot(b[i_min],h[i_min],'*r')
        plt.title(feature)
        pdf.savefig()
        plt.close()
        list_features_binary.append(feature)
        threshold = B.experiments.back_transform(feature, b[i_min])
        print('Feature: ',feature,' density peaks: ',density_peaks,' density_peaks_norm = ',density_peaks_norm,' energy = ',energy,' threshold = ',threshold, ' raw = ',b[i_min])
        #B.normalize(features = [feature,], mode = 'binary', threshold = threshold)
        #B.experiments.stretch([feature,], thrs = {'min':np.min(data_norm), 'peak_low':density_peaks_norm[0], 'threshold':b[i_min], 'peak_high':density_peaks_norm[1], 'max':np.max(data_norm)})
        B.experiments.show_histogram(pdf, list_features = [feature,])
    else:
        print('Feature: ',feature,' density peaks: ',density_peaks,' density_peaks_norm = ',density_peaks_norm,' energy = ',energy)
    B.counter += 1


print('*******List features*********: ',list_features_binary)

print('\n')
#B.fit_cluster(list_features_binary, mode = 'Unique')
#B.predict_cluster()
#ns_clusters = np.arange(2,50,1)
ns_clusters = 2

for feature_bin in list_features_binary:
    print("Working on feature ", feature_bin)
    energies = B.fit_cluster([feature_bin], ns_clusters = ns_clusters, radii = radius, mode = 'DP')
    B.predict_cluster()
    B.order_labels()
    B.save_clustering(feature_bin)
n = []    
s = []
v = []
name = 'GMP'
strategy = {'CD33':0, 'CD3':0, 'CD4':0, 'CD19':0 , 'CD8':0, 'CD34':1, 'CD38':1, 'CD45RA':1, 'CD123':0} 
versus = {'CD33':'CD45', 'CD3':'CD45', 'CD4':'CD3', 'CD8':'CD3', 'CD19':'CD45', 'CD34':'CD38', 'CD38':'CD38', 'CD45RA':'CD123', 'CD123':'CD123'} 
if (check_strategy(strategy,list_features_binary)):
    n.append(name)
    s.append(strategy)
    v.append(versus)
else:
    print("missing binarizable variables to apply strategy for ", name)
name = 'cd11b monocyte low'
strategy = {'CD33':1, 'CD3':0, 'CD4':0, 'CD8':0 , 'CD19':0, 'CD11b':0} 
versus = {'CD33':'CD45', 'CD3':'CD45', 'CD4':'CD3', 'CD8':'CD3', 'CD19':'CD45', 'CD11b':'CD45'} 
if (check_strategy(strategy,list_features_binary)):
    n.append(name)
    s.append(strategy)
    v.append(versus)
else:
    print("missing binarizable variables to apply strategy for ", name)
name = 'cd11b monocyte high' 
strategy = {'CD33':1, 'CD3':0, 'CD4':0, 'CD8':0 , 'CD19':0, 'CD11b':1} 
versus = {'CD33':'CD45', 'CD3':'CD45', 'CD4':'CD3', 'CD8':'CD3', 'CD19':'CD45', 'CD11b':'CD45'} 
if (check_strategy(strategy,list_features_binary)):
    n.append(name)
    s.append(strategy)
    v.append(versus)
else:
    print("missing binarizable variables to apply strategy for ", name)
name = 'NK'
strategy = {'CD33':0, 'CD3':0, 'CD4':0, 'CD19':0, 'CD45RA':1, 'CD38':1, 'CD8':0} 
versus = {'CD33':'CD45', 'CD3':'CD45', 'CD4':'CD3', 'CD8':'CD45RA', 'CD19':'CD45', 'CD45RA':'CD45RA', 'CD38':'CD45RA'} 
if (check_strategy(strategy,list_features_binary)):
    n.append(name)
    s.append(strategy)
    v.append(versus)
else:
    print("missing binarizable variables to apply strategy for ", name)
name = 'Plasmacytoid DC'
strategy = {'CD33':0, 'CD3':0, 'CD4':0, 'CD19':0, 'CD8':0, 'CD34':0, 'CD38':1, 'CD45RA':1, 'CD123':1} 
versus = {'CD33':'CD45', 'CD3':'CD45', 'CD4':'CD3', 'CD8':'CD3', 'CD19':'CD45', 'CD34':'CD38', 'CD38':'CD38', 'CD45RA':'CD123', 'CD123':'CD123'} 
if (check_strategy(strategy,list_features_binary)):
    n.append(name)
    s.append(strategy)
    v.append(versus)
else:
    print("missing binarizable variables to apply strategy for ", name)
name = 'Plasma cell'
strategy = {'CD33':0, 'CD3':0, 'CD4':0, 'CD19':0 , 'CD8':0} 
versus = {'CD33':'CD45', 'CD3':'CD45', 'CD4':'CD3', 'CD8':'CD3', 'CD19':'CD45', 'CD34':'CD38', 'CD38':'CD38', 'CD45RA':'CD123', 'CD123':'CD123'} 
if (check_strategy(strategy,list_features_binary)):
    n.append(name)
    s.append(strategy)
    v.append(versus)
else:
    print("missing binarizable variables to apply strategy for ", name)
name = 'Naive CD4+ T'
strategy = {'CD33':0, 'CD3':1, 'CD4':1, 'CD8':0 , 'CD19':0} 
versus = {'CD33':'CD45', 'CD3':'CD45', 'CD4':'CD3', 'CD8':'CD3', 'CD19':'CD45', 'CD34':'CD38', 'CD38':'CD38', 'CD45RA':'CD123', 'CD123':'CD123'} 
if (check_strategy(strategy,list_features_binary)):
    n.append(name)
    s.append(strategy)
    v.append(versus)
else:
    print("missing binarizable variables to apply strategy for ", name)

### INIZIO PROVA STRATEGIE
for i_strategy, strategy in enumerate(s):
    print("Working on strategy: ", n[i_strategy])
    for key in list(strategy.keys()):
        combo = B.combine_all_clustering(strategy)
        #print("COMBO: ", combo, len(combo[0]))
        if len(combo[0]) == 0:
            print("No sample corresponds to the input strategy. Stop at feature ", key)
            break;
        target_pop = np.zeros(np.shape(B.experiments.labels), dtype=bool)
        target_pop[combo[0]] = True 
        B.experiments.show_scatter(features = [key,v[i_strategy][key]], inds_inside = target_pop, pdf = pdf)

    #pdf.close()
    #### FINE PROVA STRATEGIE
    if len(combo[0])!=0:
        target_pop = np.zeros(np.shape(B.experiments.labels))
        target_pop[combo[0]] = 1 #negli indici trovati metti la classificazione a 1
        n_clusters = 2
        C = deepcopy(B)
        C.cluster.dtrajs = [target_pop] 
        C.cluster.score(n_clusters = n_clusters)
        print(C.cluster)

pdf.close()
exit()

###
energies = energies.flatten()


for i_cluster in (argrelextrema(energies, np.greater))[0][0:10]:
    n_clusters = ns_clusters[i_cluster]
    B.fit_cluster(list_features_binary, ns_clusters = n_clusters, radii = radius, mode = 'DP')
    B.predict_cluster()
    B.counter += 1

pdf.close()
exit()
