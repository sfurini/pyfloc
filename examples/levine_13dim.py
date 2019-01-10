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

strategies = {
    'HSC'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':0,  'CD45':-1, 'CD45RA':0,  'CD90':1,  'CD123':-1},
    'MPP'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':0,  'CD45':-1, 'CD45RA':0,  'CD90':0,  'CD123':-1},
    'CMP'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':2,  'CD45':-1, 'CD45RA':0,  'CD90':-1, 'CD123':0 },
    'GMP'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':2,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':0 },
    'MEP'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':2,  'CD45':-1, 'CD45RA':0,  'CD90':-1, 'CD123':1 },
    'Erythroblast'      :{'CD3':1,  'CD4':-1, 'CD8':-1, 'CD11b':-1, 'CD19':-1, 'CD20':-1, 'CD33':-1, 'CD34':-1, 'CD38':0,  'CD45':0,  'CD45RA':0,  'CD90':-1, 'CD123':-1},
    'Megakaryocyte'     :{'CD3':0,  'CD4':-1, 'CD8':-1, 'CD11b':-1, 'CD19':-1, 'CD20':-1, 'CD33':-1, 'CD34':-1, 'CD38':0,  'CD45':0,  'CD45RA':0,  'CD90':-1, 'CD123':-1},
    'Platelet'          :{'CD3':0,  'CD4':-1, 'CD8':-1, 'CD11b':-1, 'CD19':-1, 'CD20':-1, 'CD33':-1, 'CD34':-1, 'CD38':0,  'CD45':0,  'CD45RA':0,  'CD90':-1, 'CD123':-1},
    'Myelocyte'         :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':0,  'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':0,  'CD38':0,  'CD45':1,  'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'CD11blowMonocyte'  :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':0,  'CD19':0,  'CD20':-1, 'CD33':1,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'CD11bmidMonocyte'  :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':1,  'CD19':0,  'CD20':-1, 'CD33':1,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'CD11bhiMonocyte'   :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':2,  'CD19':0,  'CD20':-1, 'CD33':1,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'NK'                :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':2,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':-1},
    'plasmacytoidDC'    :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':0,  'CD38':2,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':2 },
    'PreBI'             :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'PreBII'            :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'ImmatureB'         :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'matureCD38loB'     :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'matureCD38midB'    :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':1,  'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'plasmacell'        :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'naiveCD4T'         :{'CD3':2,  'CD4':1,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':-1},        
    'matureCD4T'        :{'CD3':2,  'CD4':1,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':0,  'CD90':-1, 'CD123':-1},
    'naiveCD8T'         :{'CD3':2,  'CD4':0,  'CD8':1,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':-1},
    'matureCD8T'        :{'CD3':2,  'CD4':0,  'CD8':1,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':0,  'CD90':-1, 'CD123':-1},
}
strategies = {
    'HSC'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':0,  'CD45':-1, 'CD45RA':0,  'CD90':1,  'CD123':-1}, # checked
    'MPP'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':0,  'CD45':-1, 'CD45RA':0,  'CD90':0,  'CD123':-1}, # checked
    'CMP'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':1,  'CD45':-1, 'CD45RA':0,  'CD90':-1, 'CD123':0 },
    'GMP'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':1,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':0 },
    'MEP'               :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':1,  'CD38':1,  'CD45':-1, 'CD45RA':0,  'CD90':-1, 'CD123':1 },
    'Erythroblast'      :{'CD3':1,  'CD4':-1, 'CD8':-1, 'CD11b':-1, 'CD19':-1, 'CD20':-1, 'CD33':-1, 'CD34':-1, 'CD38':0,  'CD45':0,  'CD45RA':0,  'CD90':-1, 'CD123':-1},
    'Megakaryocyte'     :{'CD3':0,  'CD4':-1, 'CD8':-1, 'CD11b':-1, 'CD19':-1, 'CD20':-1, 'CD33':-1, 'CD34':-1, 'CD38':0,  'CD45':0,  'CD45RA':0,  'CD90':-1, 'CD123':-1},
    'Platelet'          :{'CD3':0,  'CD4':-1, 'CD8':-1, 'CD11b':-1, 'CD19':-1, 'CD20':-1, 'CD33':-1, 'CD34':-1, 'CD38':0,  'CD45':0,  'CD45RA':0,  'CD90':-1, 'CD123':-1},
    'Myelocyte'         :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':0,  'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':0,  'CD38':0,  'CD45':1,  'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'CD11blowMonocyte'  :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':0,  'CD19':0,  'CD20':-1, 'CD33':1,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'CD11bmidMonocyte'  :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':1,  'CD19':0,  'CD20':-1, 'CD33':1,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'CD11bhiMonocyte'   :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':1,  'CD19':0,  'CD20':-1, 'CD33':1,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'NK'                :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':1,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':-1},
    'plasmacytoidDC'    :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':0,  'CD38':1,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':1 },
    'PreBI'             :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'PreBII'            :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'ImmatureB'         :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'matureCD38loB'     :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'matureCD38midB'    :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':1,  'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'plasmacell'        :{'CD3':0,  'CD4':0,  'CD8':0,  'CD11b':-1, 'CD19':1,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':-1, 'CD45':-1, 'CD45RA':-1, 'CD90':-1, 'CD123':-1},
    'naiveCD4T'         :{'CD3':1,  'CD4':1,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':-1},        
    'matureCD4T'        :{'CD3':1,  'CD4':1,  'CD8':0,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':0,  'CD90':-1, 'CD123':-1},
    'naiveCD8T'         :{'CD3':1,  'CD4':0,  'CD8':1,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':1,  'CD90':-1, 'CD123':-1},
    'matureCD8T'        :{'CD3':1,  'CD4':0,  'CD8':1,  'CD11b':-1, 'CD19':0,  'CD20':-1, 'CD33':0,  'CD34':-1, 'CD38':0,  'CD45':-1, 'CD45RA':0,  'CD90':-1, 'CD123':-1},
}

B = pyfloc.PyFloc(verbose = 2, prefix = 'levine_13dim')
B.read_fcs(file_name = './data/flowc/levine_13dim.fcs', mode = 40000 )
features= ['CD3', 'CD4', 'CD8', 'CD11b', 'CD19', 'CD20', 'CD33', 'CD34', 'CD38', 'CD45', 'CD45RA', 'CD90', 'CD123']
#B.clean_samples(features = ['label',], mode = 'nan')
B.experiments.remove_outliers(features, 6.0)
B.normalize(features, mode = 'min')
#pdf = PdfPages('tmp.pdf')
#B.experiments.show_distributions(features = features, labels = B.experiments.get_data_features(['label']), pdf = pdf)
#pdf.close()
print(B)
for feature in features:
    B.show(features = [feature,])
    if feature in ['CD3','CD38','CD11b','CD123']:
        energy = B.fit_cluster([feature,], mode = 'DP', ns_clusters = 2, radii = np.linspace(0.1,0.5,10), metric = 'logarithmic')
    else:
        energy = B.fit_cluster([feature,], mode = 'DP', ns_clusters = 2, radii = np.linspace(0.1,0.5,10), metric = 'logarithmic')
    B.predict_cluster()
B.write(file_name = 'levine_13dim.pk')

#with open('levine_13dim.pk','rb') as fin:
#    B = pickle.load(fin)
#    B.counter = 0

for cell_type in strategies.keys():
    print('CELL = ',cell_type)
    B.combine_clustering(strategy = strategies[cell_type], labels = B.experiments.get_data_features(['label']))

#energy = B.fit_cluster(features, mode = 'DP', labels = B.experiments.get_data_features(['label']), ns_clusters = 23, radii = np.linspace(1.0,2.0,20), metric = 'logarithmic')
#B.predict_cluster()

exit()
