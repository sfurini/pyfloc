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
        self.counter = 0 # counter for gating steps
        self.features_synonym = {} # alternative names for features
        self.features_last_clustering = None # used to memorize the set of features used for the last clustering step, it's useful if afterwards there's a gating step
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
                                self.counter += 1
                            elif key == 'clustering': #--- Run a clustering step
                                reading_data = False
                                mode = values.split()[0].strip()
                                features = [feature.strip() for feature in ' '.join(values.split()[1:]).split(',')]
                                print('Running a clustering step for features {0:s}'.format(','.join(features)))
                                self.fit_cluster(mode, features, **parameters)
                                self.predict_cluster(**parameters)
                                ### MM
                                #print("******************************************LIST_FEATURES: ", self.experiments.get_features())
                                #exit()
                                #self.order_labels([self.experiments.get_features()[0], self.experiments.get_features()[1]]) #le prime due features, scegli quelle che vuoi
                                #self.save_clustering()
                                ###
                                self.counter += 1
                            else:   #--- Everything else is considered a definition of feature synonyms
                                self.features_synonym[key] = values
    def read_fcs(self, file_name, mode, conditions = 'undefined'):
        print('Reading data from {0:s} with mode {1:s} conditions {2:s}'.format(file_name, str(mode), conditions))
        experiment = data.Experiment(file_name, mode = mode)
        self.experiments.add_experiment(experiment, conditions)
    def clean_samples(self, features, mode):
        for feature in features:
            self.experiments.clean_samples(feature, mode)
    def remove_outliers(self, features, max_n_std = 3.0):
        self.experiments.remove_outliers(features, max_n_std, self.verbose)
    def normalize(self, features, **kwargs):
        self.experiments.normalize(features, self.verbose, **kwargs)
    def fit_cluster(self, features, mode = None, verbose = None, **kwargs):
        """
        Parameters
        ----------
        mode: str
            Clustering algorithm
        features: list
            values: str
                The features to use for clustering
        """
        #--- Use default verbosity if not defined
        if verbose is None:
            verbose = self.verbose
        #--- Get normalized data
        data_norm = self.experiments.get_data_norm_features(features)
        self.features_last_clustering = deepcopy(features)
        #--- Get reference labels, if they exist
        if 'label' in self.experiments.get_features():
            labels = [self.experiments.get_data_features(['label']).flatten(),]
        else:
            labels = []
        #--- Define mode, if None
        if mode is None:
            if self.experiments.get_n_samples() < 50000:
                mode = 'DP'
            else:
                mode = 'DPKNN'
        #--- Fit clustering mode
        if verbose > 1:
            pdf = PdfPages('{0:s}_fit_cluster_{1:d}.pdf'.format(self.prefix, self.counter))
        else:
            pdf = None
        delta_energies = None
        if mode == 'DP':
            self.cluster = cluster.DensityPeaks(trajs = [data_norm,], labels = labels, verbose = verbose)
            delta_energies = self.cluster.search_cluster_centers(pdf = pdf, **kwargs)
        elif mode == 'DPKNN':
            self.cluster = cluster.TrainingKNN(trajs = [data_norm,], labels = labels, verbose = verbose)
            self.cluster.fit(kwargs['ns_clusters'], percent = kwargs.get('percents',10.0), training_samples = kwargs.get('training_samples',10000), n_rounds = kwargs.get('n_rounds',100), pdf = pdf)
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
            pdf.close()
        return delta_energies
    def predict_cluster(self, verbose = None, **kwargs):
        #--- Use default verbosity if not defined
        if verbose is None:
            verbose = self.verbose
        #--- Run clustering
        if verbose > 1:
            pdf = PdfPages('{0:s}_cluster_{1:d}.pdf'.format(self.prefix, self.counter))
        else:
            pdf = None
        self.cluster.fit_predict(pdf = pdf, **kwargs)
        if verbose > 0:
            self.cluster.show(pdf, plot_maps = (len(self.features_last_clustering) == 2), plot_hists = (len(self.features_last_clustering) == 1))
            print(self.cluster)
        self.experiments.labels = self.cluster.dtrajs[0]
        if verbose > 1:
            self.experiments.show_distributions(features = self.features_last_clustering, pdf = pdf)
            pdf.close()
    ### MM inizio
    def save_clustering(self,feature):
        #--- adds a column in format "labels_[feature]"
        self.experiments.add_column_labels(feature) 
    def order_labels(self): #ordina dal picco piu' basso al piu' alto dell'ultimo clustering monodimensionale
        #-- orders peaks in ascending order and modifies labels so that 0 correspond to the minimum density peak and n correspond to the maximum
        new_labels = []
        cluster_analogic_sorted_temp = np.sort(np.array(self.cluster.clusters_analogic).flatten())
        cluster_analogic_sorted = deepcopy(self.cluster.clusters_analogic)
        cluster_analogic_sorted[0][0]=cluster_analogic_sorted_temp.min() 
        cluster_analogic_sorted[1][0]=cluster_analogic_sorted_temp.max()
        for i_cluster, cluster in enumerate(cluster_analogic_sorted):
            i_cluster_analogic = np.where(self.cluster.clusters_analogic == cluster)[0][0]
            new_labels.append(i_cluster_analogic)
        self.experiments.labels = np.array(self.experiments.labels)
        labels_temp = deepcopy(self.experiments.labels)
        for i_label in range(len(new_labels)):
            i_labels_temp = np.where(labels_temp == i_label)[0]
            self.experiments.labels[i_labels_temp] = new_labels[i_label]
        self.experiments.labels = list(self.experiments.labels) 
    def combine_clustering(self,feature0,feature1):
        #-- returns a dictionary which contains the indexes of the combinations of density peaks
        # i.e (0,0):[1,2,3,4] means that 1,2,3,4 are the indexes of the elements which have the lowest peak in feature0 and feature1
        return self.experiments.combine_clustering(feature0,feature1)
    def combine_all_clustering(self, dict_features):
        return self.experiments.combine_all_clustering(dict_features)
        #input: che cosa vuoi: es {'cd3' : 0 (basso) o 1 (alto), 'cd4': 0, ...}
        #solo su variabili binarizzabili
        #restituisce tutti gli indici delle cellule che hanno quelle caratteristiche
        #es per cd11b monocyte: cd33+, cd3-, cd4-, cd8-, cd19-
        #strategy = {'CD33':1, 'CD3':0, 'CD4':0, 'CD8':0 , 'CD19':0}
    ### MM fine 
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
    def write(self, file_name = None):
        if file_name is None:
            file_name = '{0:s}_{1:d}.pk'.format(self.prefix, self.counter)
        print('Dumping pyfloc class to {0:s}'.format(file_name))
        with open(file_name, 'wb') as fout:
            pickle.dump(self, fout)
    def __str__(self):
        output = self.experiments.__str__() + '\n'
        return output[:-1]
