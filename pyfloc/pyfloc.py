#!/usr/bin/env python

import sys
import pickle
import functools
from copy import deepcopy
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import data
import cluster
import contour

class PyFloc(object):
    """
    Attributes
    ----------
    experiments: Collection
    prefix: str
        Prefix for output files
    counter_gating: int
        Internal variable to keep track of the gating steps performed
    counter_clustering: int
        Internal variable to keep track of the clustering steps performed
    features_synonym:   dict
        key:    str
            Name of the feature
        value:  str
            Alternative name
    verbose:    int
        0:  
        1:  
        2:  + all the figures
    """
    def __init__(self, prefix = 'pyfloc', verbose = 0):
        self.prefix = prefix # all output files will start like this
        self.experiments = data.Collection() # initialize a new collection of experiments
        self.counter_gating = 0 # counter for gating steps
        self.counter_clustering = 0 # counter for clustering steps
        self.features_synonym = {} # alternative names for features
        self.features_last_clustering = None # used to memorize the set of features used for the last clustering step, it's useful if afterwards there's a gating step
        self.last_gate = None # used to memorize the last contour
        self.gate_mode = None # used to memorize if gating was in linear or log scale
        self.verbose = verbose # how much output is produced in terms of figures and text
    def read_input(self, file_input):
        input_format = 'fcs' # default input file format - this is used to check that all data are read from fsc OR pk files
        reading_data = True # reading data flag - this is used to check that first all the data are read and then the anaylyses start
        parameters = {} # these are parameters passed to gating/clustering functions
        with open(file_input,'rt') as fin:
            for l in fin.readlines():
                lc = l.strip()
                if lc:
                    if lc[0] != '#':
                        lf = lc.split('=')
                        if len(lf) > 1:
                            key = lf[0].strip()
                            values = lf[1].strip()
                            if key[0:2] == '__': #--- Parameters for gating/clustering algorithms
                                    try:
                                        parameters[key[2:]] = int(values)
                                    except:
                                        try:
                                            parameters[key[2:]] = float(values)
                                        except:
                                            try:
                                                parameters[key[2:]] = [float(value) for value in values.split(',')]
                                            except:
                                                parameters[key[2:]] = values
                            elif key == 'file_fcs': #--- Read data from fcs file
                                if input_format != 'fcs':
                                    raise ValueError('ERROR: wrong format in file {0:s}'.format(file_input))
                                if not reading_data:
                                    raise ValueError('ERROR: wrong format in file {0:s}'.format(file_input))
                                file_name = values.split()[0].strip()
                                read_mode = values.split()[1].strip()
                                conditions = ','.join([s.strip() for s in values.split()[2:]])
                                if len(conditions) == 0:
                                    conditions = 'unknown'
                                self.read_fcs(file_name, read_mode, conditions)
                            elif key == 'pk': #--- Read experiments from pk file
                                if self.experiments.get_n_experiments():
                                    raise ValueError('ERROR: wrong format in file {0:s}'.format(file_input))
                                if not reading_data:
                                    raise ValueError('ERROR: wrong format in file {0:s}'.format(file_input))
                                input_format = 'pk'
                                file_name = values.split()[0].strip()
                                print('Reading data from {0:s}'.format(file_name))
                                with open(file_name,'rb') as fin:
                                    self.experiments = pickle.load(fin)
                            elif key == 'compensate': #--- Compensate data
                                reading_data = False
                                if values.lower() == 'true':
                                    if input_format != 'fcs':
                                        raise ValueError('ERROR: wrong format in file {0:s}'.format(file_input))
                                    self.experiments.compensate()
                            elif key == 'clean': #--- Remove samples from experiments
                                reading_data = False
                                feature = values.split()[0]
                                if feature == 'all':
                                    features = self.experiments.get_features()
                                else:
                                    features = [feature,]
                                remove = values.split()[1]
                                self.clean_samples(features, remove)
                            elif key == 'remove_outliers': #--- Remove outliers
                                reading_data = False
                                features = [feature.strip() for feature in values.split(',')]
                                self.remove_outliers(features, parameters.get('max_n_std', 3.0))
                            elif key == 'features_normalize': #--- Normalize data
                                reading_data = False
                                features = [feature.strip() for feature in values.split(',')]
                                self.normalize(features, **parameters)
                            elif key == 'gating': #--- Run a gating step
                                reading_data = False
                                prob_target = float(values.split()[0].strip())
                                mode = values.split()[1].strip()
                                if len(values.split()) > 2:
                                    clusters_2_keep = [int(s.strip()) for s in ' '.join(values.split()[2:]).split(',')]
                                else:
                                    clusters_2_keep = []
                                self.draw_gate(prob_target, mode, clusters_2_keep)
                                self.apply_gate()
                                self.counter_gating += 1
                            elif key == 'clustering': #--- Run a clustering step
                                reading_data = False
                                mode = values.split()[0].strip()
                                features = [feature.strip() for feature in ' '.join(values.split()[1:]).split(',')]
                                print('Running a clustering step for features {0:s}'.format(','.join(features)))
                                self.fit_cluster(mode, features, **parameters)
                                self.predict_cluster(**parameters)
                                self.counter_clustering += 1
                            else:   #--- Everything else is considered a definition of feature synonyms
                                self.features_synonym[key] = values
    def read_fcs(self, file_name, read_mode, conditions = 'undefined'):
        print('Reading data from {0:s} with mode {1:s} conditions {2:s}'.format(file_name, read_mode, conditions))
        experiment = data.Experiment(file_name, mode = read_mode)
        self.experiments.add_experiment(experiment, conditions)
    def clean_samples(self, features, remove):
        for feature in features:
            self.experiments.clean_samples(feature, remove)
    def remove_outliers(self, features, max_n_std = 3.0):
        self.experiments.remove_outliers(features, max_n_std)
    def normalize(self, features, **kwargs):
        self.experiments.normalize(features, **kwargs)
    def fit_cluster(self, mode, features, **kwargs):
        """
        Parameters
        ----------
        mode: str
            Clustering algorithm
        features: list
            values: str
                The features to use for clustering
        """
        #--- Get normalized data
        data_norm = self.experiments.get_data_norm_features(features)
        self.features_last_clustering = deepcopy(features)
        #--- Get reference labels, if they exist
        if 'label' in self.experiments.get_features():
            labels = self.experiments.get_data_features(['label']).flatten()
        else:
            labels = None
        #--- Fit clustering mode
        if self.verbose > 1:
            pdf = PdfPages('{0:s}_fit_cluster_{1:d}.pdf'.format(self.prefix, self.counter_clustering))
        else:
            pdf = None
        if mode == 'DensityPeaks':
            self.cluster = cluster.DensityPeaks(trajs = [data_norm,], labels = [labels,])
            self.cluster.search_cluster_centers(pdf = pdf, **kwargs)
        elif mode == 'Kmeans':
            self.cluster = cluster.Kmeans(trajs = [data_norm,], labels = [labels,])
        else:
            raise NotImplemented('ERROR: mode {0:s} not implemented'.format(mode))
        if self.verbose > 1:
            pdf.close()
    def predict_cluster(self, **kwargs):
        #--- Run clustering
        if self.verbose > 1:
            pdf = PdfPages('{0:s}_cluster_{1:d}.pdf'.format(self.prefix, self.counter_clustering))
        else:
            pdf = None
        self.cluster.fit_predict(pdf = pdf, **kwargs)
        self.cluster.show(pdf, plot_maps = (len(self.features_last_clustering) == 2))
        print(self.cluster)
        self.experiments.labels = self.cluster.dtrajs[0]
        if self.verbose > 1:
            pdf.close()
    def draw_gate(self, prob_target, mode, clusters_2_keep = []):
        if (len(self.features_last_clustering) != 2) and (prob_target > 0):
            raise ValueError('ERROR: gating works only with 2 features')
        if not clusters_2_keep:
            input_str = input('Look at last clustering file and choose the cluster(s) to keep (write cluster indexes separated by commas): ')
            clusters_2_keep = [int(s.strip()) for s in input_str.split(',')]
        self.gate_mode = mode
        #--- Delete samples based on clustering
        print('Number of samples before removing clusters: {0:d}'.format(self.experiments.get_n_samples()))
        print('Keeping samples from clusters: ',clusters_2_keep)
        inds_2_delete = [i_sample for i_sample, label in enumerate(self.experiments.labels) if label not in clusters_2_keep]
        self.experiments.delete_samples(inds_2_delete)
        print('Number of samples after removing clusters: {0:d}'.format(self.experiments.get_n_samples()))
        #--- Get the data
        data = self.experiments.get_data_features(self.features_last_clustering)
        lin_upper = 1e-3
        lin_lower = -1e-3
        if self.gate_mode == 'log-log' or self.gate_mode == 'log-lin':
            inds = data[:,0] > lin_upper
            data[inds,0] = np.log10(data[inds,0])
            inds = data[:,0] < lin_lower
            data[inds,0] = -np.log10(-data[inds,0])
        if self.gate_mode == 'log-log' or self.gate_mode == 'lin-log':
            inds = data[:,1] > lin_upper
            data[inds,1] = np.log10(data[inds,1])
            inds = data[:,1] < lin_lower
            data[inds,1] = -np.log10(-data[inds,1])
        elif self.gate_mode != 'lin-lin':
            raise ValueError('ERROR: mode {0:s} is not implemented'.format(self.gate_mode))
        #--- Draw the contour
        if prob_target > 0.0: #--- Define contour
            self.last_gate = contour.Contour(data, n_gaussians = 20, prob_target = prob_target, n_bins = [100,100])
            self.last_gate.run(n_points = 20, max_iter = 10000, stride_show = 1000, tol = 1e-2)
    def apply_gate(self):
        if self.last_gate is None:
            print('ERROR: First run draw_gate')
            return
        print('Number of samples before gating: {0:d}'.format(self.experiments.get_n_samples()))
        if self.verbose > 1:
            pdf = PdfPages('{0:s}_gating_{1:d}.pdf'.format(self.prefix,self.counter_gating))
        else:
            pdf = None
        inds_inside = self.last_gate.get_index_inside_polygon()
        inds_outside = np.logical_not(inds_inside)
        contour = [np.array(self.last_gate.xc), np.array(self.last_gate.yc)]
        lin_upper = 1e-3
        lin_lower = -1e-3
        if self.gate_mode == 'log-log' or self.gate_mode == 'log-lin':
            inds = contour[0] > lin_upper
            contour[0][inds] = np.power(10, contour[0][inds])
            inds = contour[0] < lin_lower
            contour[0][inds] = -np.power(10,-contour[0][inds])
        if self.gate_mode == 'log-log' or self.gate_mode == 'lin-log':
            inds = contour[1] > lin_upper
            contour[1][inds] = np.power(10, contour[1][inds])
            inds = contour[1] < lin_lower
            contour[1][inds] = -np.power(10,-contour[1][inds])
        elif self.gate_mode != 'lin-lin':
            raise ValueError('ERROR: mode {0:s} is not implemented'.format(self.gate_mode))
        self.experiments.show_scatter(self.features_last_clustering, features_synonym = self.features_synonym, contour = contour, pdf = pdf)
        self.experiments.delete_samples(inds_outside)
        print('Number of samples after gating: {0:d}'.format(self.experiments.get_n_samples()))
        if self.verbose > 1:
            pdf.close()
    def write(self, file_name):
        print('Dumping data to {0:s}'.format(file_name))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.experiments, fout)
    def __str__(self):
        output = self.experiments.__str__() + '\n'
        return output[:-1]
