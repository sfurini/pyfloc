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
    prefix: str
        Prefix for output files
    experiments: object of class Collection
    counter: int
        Internal variable to keep track of the steps performed
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
        self.prefix = prefix # all output files start like this
        self.experiments = data.Collection() # initialize a new collection of experiments
        self.counter = 0 # counter for gating steps
        self.features_synonym = {} # alternative names for features
        self.features_last_clustering = None # used to memorize the set of features used for the last clustering step
        self.cluster = None # used to memorize the last clustering object
        self.last_gate = None # used to memorize the last contour
        self.clusters_2_keep = None
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
                                mode = values.split()[1].strip()
                                conditions = ','.join([s.strip() for s in values.split()[2:]])
                                if len(conditions) == 0:
                                    conditions = 'unknown'
                                self.read_fcs(file_name, mode, conditions)
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
                                mode = values.split()[1]
                                self.clean_samples(features, mode)
                            elif key == 'remove_outliers': #--- Remove outliers
                                reading_data = False
                                features = [feature.strip() for feature in values.split(',')]
                                self.remove_outliers(features, parameters.get('max_n_std', 3.0))
                            elif key == 'features_normalize': #--- Normalize data
                                reading_data = False
                                features = [feature.strip() for feature in values.split(',')]
                                self.experiments.normalize(features, self.verbose, **parameters)
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
                                self.counter += 1
                            elif key == 'clustering': #--- Run a clustering step
                                reading_data = False
                                mode = values.split()[0].strip()
                                features = [feature.strip() for feature in ' '.join(values.split()[1:]).split(',')]
                                print('Running a clustering step for features {0:s}'.format(','.join(features)))
                                self.fit_cluster(mode, features, **parameters)
                                self.predict_cluster(**parameters)
                                self.counter += 1
                            else:   #--- Everything else is considered a definition of feature synonyms
                                self.features_synonym[key] = values
    def read_fcs(self, file_name, mode, conditions = 'unknown'):
        print('Reading data from {0:s} with reading mode {1:s} - condition: {2:s}'.format(file_name, str(mode), conditions))
        experiment = data.Experiment(file_name, mode = mode)
        self.experiments.add_experiment(experiment, conditions)
    def clean_samples(self, features, mode):
        for feature in features:
            self.experiments.clean_samples(feature, mode)
    def remove_outliers(self, features, max_n_std = 3.0):
        self.experiments.remove_outliers(features, max_n_std, self.verbose)
    def fit_cluster(self, features, mode, labels = [], fuzzy = False, verbose = None, **kwargs):
        """
        Parameters
        ----------
        features: list of str
        mode: str
            Clustering algorithm
        labels: list / np.ndarray
        verbose: int
            0-1: none
            >1: figures in pdf
        """
        #--- Use default verbosity, if not defined
        if verbose is None:
            verbose = self.verbose
        #--- Get normalized data
        data_norm = self.experiments.get_data_norm_features(features)
        self.features_last_clustering = deepcopy(features)
        #--- Get reference labels, if they exist
        if  len(labels):
            if len(labels) != self.experiments.get_n_samples():
                raise ValueError('ERROR: wrong number of labels')
        if isinstance(labels, np.ndarray):
            labels = [labels.flatten(),]
        #--- Fit clustering mode
        if verbose > 1:
            pdf = PdfPages('{0:s}_{1:d}_fit_cluster.pdf'.format(self.prefix, self.counter))
        else:
            pdf = None
        delta_energies = None
        if mode == 'DP':
            self.cluster = cluster.DensityPeaks(trajs = [data_norm,], labels = labels, fuzzy = fuzzy, verbose = verbose)
            delta_energies = self.cluster.search_cluster_centers(pdf = pdf, **kwargs)
        elif mode == 'DPKNN':
            self.cluster = cluster.TrainingKNN(trajs = [data_norm,], labels = labels, verbose = verbose)
            self.cluster.fit(pdf = pdf, **kwargs)
        elif mode == 'Unique':
            data_norm = data_norm.astype('int')
            self.cluster = cluster.Unique(trajs = [data_norm,], labels = labels, verbose = verbose)
            self.cluster.fit()
        elif mode == 'Kmeans':
            self.cluster = cluster.Kmeans(trajs = [data_norm,], labels = labels, verbose = verbose)
            self.cluster.fit(kwargs['ns_clusters'], pdf = pdf)
        else:
            raise NotImplementedError('ERROR: mode {0:s} not implemented'.format(mode))
        if verbose > 1:
            self.counter += 1
            pdf.close()
        return delta_energies
    def predict_cluster(self, verbose = None, **kwargs):
        """
        Parameters
        ----------
        verbose: int
            0-1: none
            >1: (output of clustering) + (pdf figures of clustering)
        """
        #--- Use default verbosity, if not defined
        if verbose is None:
            verbose = self.verbose
        #--- Open pdf file, if needed
        if verbose > 1:
            pdf = PdfPages('{0:s}_{1:d}_predict_cluster.pdf'.format(self.prefix, self.counter))
        else:
            pdf = None
        #--- Run clustering
        self.cluster.fit_predict(pdf = pdf, **kwargs)
        #--- Save the results of clustering
        self.experiments.add_feature('label_'+'_'.join(self.features_last_clustering), self.cluster.dtrajs[0]) # this is useful when combining different clustering strategies (e.g.: several 1D clustering to mimik a gating strategy)
        if self.cluster.fuzzy: # if fuzzy clustering, save also the probabilities
            for i_cluster in range(self.cluster.n_clusters):
                if len(self.cluster.probs) != 1:
                    raise ValueError('ERROR: wrong dimension for self.cluster.probs')
                self.experiments.add_feature('prob_{0:d}_'.format(i_cluster)+'_'.join(self.features_last_clustering), self.cluster.probs[0][:,i_cluster]) 
        #--- Plotting and outputs
        if verbose > 1:
            self.cluster.show(pdf, plot_maps = (len(self.features_last_clustering) == 2), plot_hists = (len(self.features_last_clustering) == 1))
            print(self.cluster)
            self.counter += 1
            pdf.close()
    def combine_clustering(self, strategy, labels = [], fuzzy = False, verbose = None):
        """
        Parameters
        ----------
        strategy: dict
            key: str
            value: int
        """
        #--- Use default verbosity, if not defined
        if verbose is None:
            verbose = self.verbose
        #--- Check reference labels, if they exist
        if  len(labels):
            if len(labels) != self.experiments.get_n_samples():
                raise ValueError('ERROR: wrong number of labels {0:d} Vs {1:d}'.format(len(labels), self.experiments.get_n_samples()))
        if isinstance(labels, np.ndarray):
            labels = [labels.flatten(),]
        #--- Retrieving data from previous clustering
        clustering_features = []
        target = []
        if fuzzy:
            probs = np.ones((self.experiments.get_n_samples(),2))
        else:
            probs = None
        for feature in strategy.keys():
            if strategy[feature] >= 0: # -1 means to not consider that feature
                clustering_features.append('label_{0:s}'.format(feature))
                target.append(int(strategy[feature]))
                if fuzzy:
                    prob = self.experiments.get_data_features(['prob_{0:d}_{1:s}'.format(target[-1], feature),]) # probability of the target cluster
                    probs[:,1] *= prob.flatten()
        data = self.experiments.get_data_features(clustering_features)
        if fuzzy:
            probs[:,0] = 1.0 - probs[:,1]
            probs = [probs,] # this is beacuase of the particular format of the class Cluster
        #--- Convert into discrete trajectory: 1 == target / 0 != target
        traj = []
        for i_sample in range(self.experiments.get_n_samples()):
            if (list(data[i_sample].astype(int))) == target:
                traj.append(1)
            else:
                traj.append(0)
        #--- Running clustering
        self.cluster = cluster.Unique(trajs = [np.array(traj).astype(int).reshape(-1,1),], labels = labels, list_samples = [[0,],[1,]], probs = probs, verbose = verbose)
        self.cluster.fit_predict()
        print(self.cluster)
    def discover_cells(self, features, labels = [], verbose = None):
        """
        Parameters
        ----------
        strategy: dict
            key: str
            value: int
        """
        #--- Use default verbosity, if not defined
        if verbose is None:
            verbose = self.verbose
        #--- Check reference labels, if they exist
        if  len(labels):
            if len(labels) != self.experiments.get_n_samples():
                raise ValueError('ERROR: wrong number of labels {0:d} Vs {1:d}'.format(len(labels), self.experiments.get_n_samples()))
        if isinstance(labels, np.ndarray):
            labels = [labels.flatten(),]
        #--- Retrieving data from previous clustering
        clustering_features = []
        for feature in features:
            clustering_features.append('label_{0:s}'.format(feature))
        data = self.experiments.get_data_features(clustering_features)
        #--- Running clustering
        self.cluster = cluster.Unique(trajs = [data,], labels = labels, verbose = verbose)
        self.cluster.fit_predict()
        #--- Plotting and outputs
        print(self.cluster)
        if verbose > 1:
            pdf = PdfPages('{0:s}_{1:d}_discover_cells.pdf'.format(self.prefix, self.counter))
            self.experiments.show_distributions(features = features, labels = self.cluster.dtrajs_merged, discrete_data = True, pdf = pdf)
            self.counter += 1
            pdf.close()
    def draw_gate(self, target, mode = 'cherry', clusters_2_keep = [], verbose = None):
        #--- Use default verbosity if not defined
        if verbose is None:
            verbose = self.verbose
        #--- Check parameters
        if (len(self.features_last_clustering) != 2) and (prob_target > 0):
            raise ValueError('ERROR: gating works only with 2 features')
        if not clusters_2_keep:
            input_str = input('Look at last clustering file and choose the cluster(s) to keep (write cluster indexes separated by commas): ')
            clusters_2_keep = [int(s.strip()) for s in input_str.split(',')]
        self.clusters_2_keep = clusters_2_keep
        #--- Get the data
        data = self.experiments.get_data_norm_features(self.features_last_clustering)
        inds_2_keep = [i_sample for i_sample, label in enumerate(self.experiments.labels) if label in self.clusters_2_keep]
        inds_2_delete = [i_sample for i_sample, label in enumerate(self.experiments.labels) if label not in self.clusters_2_keep]
        if len(inds_2_delete):
            outside_data = data[inds_2_delete,:]
        else:
            outside_data = None
        #--- Draw the contour
        if target > 0.0: #--- Define contour
            if mode == 'mgaussian':
                self.last_gate = contour.Contour(data, n_gaussians = 20, prob_target = target, n_bins = [100,100])
                self.last_gate.run(n_points = 20, max_iter = 10000, stride_show = 1000, tol = 1e-2)
            elif mode == 'cherry':
                self.last_gate = contour.Cherry(data = data[inds_2_keep,:], outside_data = outside_data, n_bins = [100, 100], verbose = verbose)
                self.last_gate.run(prob_target = target, starting_point = None, exclude_borders = False, mode='above')
            else:
                raise NotImplementedError('ERROR: mode {0:s} does not exist'.format(mode))
    def apply_gate(self, verbose = None):
        #--- Use default verbosity if not defined
        if verbose is None:
            verbose = self.verbose
        #--- Check parameters
        if (self.last_gate is None) or (self.clusters_2_keep is None):
            print('ERROR: First run draw_gate')
            return
        #--- Delete samples based on clustering
        #print('Number of samples before removing clusters: {0:d}'.format(self.experiments.get_n_samples()))
        #print('Keeping samples from clusters: ',self.clusters_2_keep)
        #inds_2_delete = [i_sample for i_sample, label in enumerate(self.experiments.labels) if label not in self.clusters_2_keep]
        #self.experiments.delete_samples(inds_2_delete, self.verbose)
        #print('Number of samples after removing clusters: {0:d}'.format(self.experiments.get_n_samples()))
        #--- Delete samples based on gating
        print('Number of samples before gating: {0:d}'.format(self.experiments.get_n_samples()))
        if self.verbose > 1:
            for conditions in self.experiments.get_conditions():
                print('\tconditions {0:s} = {1:d}'.format(conditions,self.experiments.get_n_samples(conditions)))
        self.last_gate.get_polygon_refined() # update the contour, if it was manually modified
        inds_inside = self.last_gate.get_index_inside_polygon(data = self.experiments.get_data_norm_features(self.features_last_clustering))
        inds_outside = np.logical_not(inds_inside)
        contour = [np.array(self.last_gate.xc), np.array(self.last_gate.yc)]
        if self.verbose > 1:
            pdf = PdfPages('{0:s}_gating_{1:d}.pdf'.format(self.prefix,self.counter))
            self.experiments.show_scatter(self.features_last_clustering, features_synonym = self.features_synonym, contour = contour, mode = 'density', inds_inside = inds_inside, pdf = pdf)
            pdf.close()
        self.experiments.delete_samples(inds_outside)
        print('Number of samples after gating: {0:d}'.format(self.experiments.get_n_samples()))
        if self.verbose > 1:
            for conditions in self.experiments.get_conditions():
                print('\tconditions {0:s} = {1:d}'.format(conditions,self.experiments.get_n_samples(conditions)))
    def show(self, *args, **kwargs):
        pdf = PdfPages('{0:s}_{1:d}.pdf'.format(self.prefix,self.counter))
        if 'features' not in kwargs.keys():
            raise ValueError('ERROR: missing features for show method')
        if len(kwargs['features']) == 1:
            self.experiments.show_histogram(*args, **kwargs, pdf = pdf)
        if len(kwargs['features']) == 2:
            self.experiments.show(*args, **kwargs, pdf = pdf)
        self.counter += 1
        pdf.close()
    def write(self, file_name = None):
        if file_name is None:
            file_name = '{0:s}_{1:d}.pk'.format(self.prefix, self.counter)
        print('Dumping pyfloc class to {0:s}'.format(file_name))
        with open(file_name, 'wb') as fout:
            pickle.dump(self, fout)
    def __str__(self):
        output = self.experiments.__str__() + '\n'
        return output[:-1]
