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
from FlowCytometryTools import FCMeasurement   
import logicleScale
import graphics
import settings

plt.switch_backend('agg')

np.set_printoptions(linewidth = np.inf)
print = functools.partial(print, flush=True)

def scatter_plot(ax, data, data_colors, contour):
    if isinstance(data_colors,str):
        if data_colors == 'density':
            from scipy.stats import binned_statistic_2d
            H, xe, ye, ix_iy = binned_statistic_2d(data[:,0], data[:,1], None, statistic = 'count', bins = 100, range = [[data[:,0].min(), data[:,0].max()],[data[:,1].min(), data[:,1].max()]], expand_binnumbers = True)
            ix_iy -= 1
            data_colors = H[ix_iy[0,:], ix_iy[1,:]]
            data_colors = np.log10(data_colors)
            ax.scatter(data[:,0], data[:,1], marker = ',', s = 1.0, c = data_colors, cmap = 'inferno')
        else:
            raise ValueError('ERROR: wrong colouring mode in scatter plot')
    else:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.plot(data[:,0], data[:,1],',k',markersize = 1.0, label = '')
        for data_color in set(data_colors):
            inds_data_color = (data_colors == data_color)
            ax.plot(data[inds_data_color,0], data[inds_data_color,1], '.', markersize = 2.0, color = settings.colors[data_color%len(settings.colors)], label = str(data_color))
    if contour is not None:
        ax.plot(contour[0], contour[1], '--r')

class Collection(object):
    """
    Set of FCS experiments

    Attributes
    ----------
    conditions: list of str
        These are the conditions of the experiments
        e.g.: ['controls', 'cases']
    experiments: list of object of class Experiment
    normalize_mode: dict
        key: str
            Feature names
        values: str
            The method used to normalize the corresponding feature
    normalize_parameters: dict
        key: str
            Feature names
        values: list of float
            These values have different meaning depending on the strategy used to normalize the features
    fis: list
        Variables used for interactive figures
    """
    def __init__(self):
        self.conditions = []
        self.experiments = []
        self.normalize_mode = {}
        self.normalize_parameters = {}
        self.fis = None
    def add_experiment(self, experiment, condition, labels = None):
        """
	Parameters
	----------
        experiment: object of class Experiment
        condition: str
        labels: np.ndarray
            If provided, it needs to be an array with length equal to the number of samples
            nan values are converted to -1
            If not provided, all labels are set to -1
        """
        self.conditions.append(condition)
        self.experiments.append(experiment)
    def add_feature(self, feature, data):
        """
        Add a new feature to all the experiments

        Parameters
        ----------
        feature: str
            The name of the feature to add
        data: np.ndarray / list
            The values for the new features
        """
        for i_experiment, experiment in enumerate(self.experiments):
            i_start, i_end = self.get_boundaries(i_experiment)
            experiment.add_feature(feature, data[i_start:i_end])
    def get_n_conditions(self):
        return len(set(self.conditions))
    def get_conditions(self):
        set_conditions = set(self.conditions)
        return list(set_conditions)
    def get_n_experiments(self, conditions = None):
        """
        Count the number of experiments

        Parameters
        ----------
        conditions: list
            If provided only the experiments of these conditions are counted

        Returns
        -------
        int
            The number of experiments
        """
        if conditions is None:
            return len(self.experiments)
        if not isinstance(conditions, list):
            conditions = [conditions,]
        n_experiments = 0
        for i_experiment, condition in enumerate(self.conditions):
            if condition in conditions:
                n_experiments += 1
        return n_experiments
    def get_n_samples(self, conditions = None):
        """
        Count the number of samples

        Parameters
        ----------
        conditions: list
            If provided only the samples of these conditions are counted

        Returns
        -------
        int
            The number of samples
        """
        if conditions is not None:
            if not isinstance(conditions, list):
                conditions = [conditions,]
        n_samples = 0
        for i_experiment, experiment in enumerate(self.experiments):
            if (conditions is None) or (self.conditions[i_experiment] in conditions): 
                n_samples += experiment.get_n_samples()
        return n_samples
    def get_features(self):
        """
        Return
        ------
        list
            Features common to all the experiments
        """
        for i_experiment, experiment in enumerate(self.experiments):
            if i_experiment == 0:
                common_features = set(experiment.get_features())
            else:
                common_features &= set(experiment.get_features())
        return common_features
    def get_data_features(self, features):
        """
        Parameters
        ----------
        features: list
            The features to extract

        Return
        ------
        np.array
            Data for the requested feature
        """
        data = np.empty((0,len(features)))
        for experiment in self.experiments:
            data_experiment = experiment.get_data_features(features)
            data = np.vstack((data, data_experiment))
        return data
    def get_data_norm_features(self, features):
        """
        Parameters
        ----------
        features: list
            The features to extract

        Return
        ------
        np.array
            Normalized data for the requested features
        """
        data_norm = np.empty((0,len(features)))
        for experiment in self.experiments:
            if experiment.has_norm(features):
                data_norm_experiment = experiment.get_data_norm_features(features).reshape(-1,len(features))
            else:
                data_norm_experiment = np.empty((experiment.get_n_samples(), len(features)))
                data_norm_experiment[:] = np.nan
            data_norm = np.vstack((data_norm, data_norm_experiment))
        return data_norm
    def get_something_feature(self, feature, function, norm = False):
        if norm:
            data = self.get_data_norm_features([feature])
        else:
            data = self.get_data_features([feature])
        if (len(data) == 0):
            return np.nan
        return function(data)
    def get_min_feature(self, feature):
        return self.get_something_feature(feature, np.nanmin)
    def get_max_feature(self, feature):
        return self.get_something_feature(feature, np.nanmax)
    def get_mean_feature(self, feature):
        return self.get_something_feature(feature, np.nanmean)
    def get_std_feature(self, feature):
        return self.get_something_feature(feature, np.nanstd)
    def get_min_norm_feature(self, feature):
        return self.get_something_feature(feature, np.nanmin, norm = True)
    def get_max_norm_feature(self, feature):
        return self.get_something_feature(feature, np.nanmax, norm = True)
    def get_mean_norm_feature(self, feature):
        return self.get_something_feature(feature, np.nanmean, norm = True)
    def get_std_norm_feature(self, feature):
        return self.get_something_feature(feature, np.nanstd, norm = True)
    def get_boundaries(self, i_experiment):
        """
        Parameters
        ----------
        i_experiment: int
            Index of the object Experiment in the list self.experiments

        Return
        ------
        ind_start: int
            The index of the first element of the experiment i_experiment
        ind_end: int
            The index of the last element of the experiment i_experiment
        """
        ind_start = 0
        ind_end = self.experiments[0].get_n_samples()
        for i in range(1,i_experiment+1):
            ind_start += self.experiments[i-1].get_n_samples()
            ind_end += self.experiments[i].get_n_samples()
        return ind_start, ind_end
    def get_indexes_conditions(self, conditions):
        """
        Parameters
        ----------
        conditions: list / str
            
        Return
        ------
        list
            Indexes of samples belonging to the defined conditions
        """
        inds = []
        if not isinstance(conditions, list):
            conditions = [conditions,]
        for i_experiment, experiment in enumerate(self.experiments):
            if self.conditions[i_experiment] in conditions:
                ind_start, ind_end = self.get_boundaries(i_experiment)
                inds.extend(range(ind_start, ind_end))
        return inds
    def get_indexes_experiment(self, inds, i_experiment):
        """
        Return
        ------
        list
            Indexes in inds that belongs to experiment i_experiment in the framework of i_experiment
        """
        ind_start, ind_end = self.get_boundaries(i_experiment)
        #print('Experiment {0:d} goes from sample {1:d} to sample {2:d}'.format(i_experiment, ind_start, ind_end))
        inds_experiment = []
        for ind in inds:
            if (ind >= ind_start) and (ind < ind_end):
                inds_experiment.append(ind-ind_start)
        return inds_experiment
    def compensate(self):
        for i_experiment, experiment in enumerate(self.experiments):
            print('Running compensation for experiment {0:d}'.format(i_experiment))
            experiment.compensate()
    def normalize(self, features, verbose = 0, **kwargs):
        """
        Parameters
        ----------
        features list of str
            Name of the features to normalize
        """
        if 'mode' not in kwargs:
            kwargs['mode'] = 'min'
        print('Running feature normalization with mode {0:s}'.format(kwargs['mode']))
        for feature in features:
            if kwargs['mode'] == 'min':
                #self.normalize_parameters[feature] = [self.get_min_feature(feature), self.get_mean_feature(feature)]
                self.normalize_parameters[feature] = [self.get_min_feature(feature), settings.numerical_precision]
            elif kwargs['mode'] == 'min_max':
                min_feature = self.get_min_feature(feature)
                max_feature = self.get_max_feature(feature)
                self.normalize_parameters[feature] = [min_feature, max_feature]
            elif kwargs['mode'] == 'mean_std':
                mean_feature = self.get_mean_feature(feature)
                std_feature = self.get_std_feature(feature)
                self.normalize_parameters[feature] = [mean_feature, std_feature]
            elif kwargs['mode'] == 'log10':
                self.normalize_parameters[feature] = [self.get_min_feature(feature), 0.0 ]#settings.numerical_precision]
                self.normalize_parameters[feature] = [0.0, 0.0 ]#settings.numerical_precision]
            elif kwargs['mode'] == 'sqrt':
                self.normalize_parameters[feature] = self.get_min_feature(feature)
            elif kwargs['mode'] == 'arcsinh':
                self.normalize_parameters[feature] = [kwargs.get('bias',0.0), kwargs.get('factor',5.0)]
                if self.normalize_parameters[feature][0] == 'min':
                    min_feature = self.get_min_feature(feature)
                    self.normalize_parameters[feature][0] = min_feature
            elif kwargs['mode'] == 'sigmoid':
                self.normalize_parameters[feature] = [kwargs.get('bias',0.0),kwargs.get('factor',1.0)]
            elif kwargs['mode'] == 'binary':
                self.normalize_parameters[feature] = [kwargs['threshold'],]
            elif kwargs['mode'] == 'logicle':
                data_feature = self.get_data_features([feature])
                L = logicleScale.LogicleScale(data_feature)
                L.calculate_T_M_A_r()
                L.calculate_p_W()
                self.normalize_parameters[feature] = [L.T, L.M, L.A, L.p, L.W] 
            else:
                raise ValueError('ERROR: '+kwargs['mode']+' is not a normalization mode')
            self.normalize_mode[feature] = kwargs['mode']    
            for i_experiment, experiment in enumerate(self.experiments):
                if (verbose > 0):
                    print('Running normalization for feature {0:s} in experiment {1:d} with mode {2:s}'.format(feature, i_experiment,kwargs.get('mode','min')))
                experiment.normalize(feature, self.transform)
    def transform(self, feature, data):
        """
        Parameters
        ----------
        feature: str
            Name of the feature
        data: np.ndarray
            One dimensional array with data along that feature
        """
        if self.normalize_mode[feature] == 'min':
            return data - self.normalize_parameters[feature][0] + self.normalize_parameters[feature][1]
        elif self.normalize_mode[feature] == 'min_max':
            return (data - self.normalize_parameters[feature][0]) / (self.normalize_parameters[feature][1] - self.normalize_parameters[feature][0]) 
        elif self.normalize_mode[feature] == 'mean_std':
            return (data - self.normalize_parameters[feature][0]) / self.normalize_parameters[feature][1]
        elif self.normalize_mode[feature] == 'arcsinh':
            return np.arcsinh((data - self.normalize_parameters[feature][0])/self.normalize_parameters[feature][1])
        elif self.normalize_mode[feature] == 'log10':
            dummy = (data - self.normalize_parameters[feature][0])
            return np.log10(dummy + self.normalize_parameters[feature][1])
        elif self.normalize_mode[feature] == 'sqrt':
            dummy = data - self.normalize_parameters[feature]
            return np.sqrt(dummy)
        elif self.normalize_mode[feature] == 'sigmoid':
            dummy = (data - self.normalize_parameters[feature][0])/self.normalize_parameters[feature][1]
            return 1.0 / ( 1.0 + np.exp(-dummy) )
        elif self.normalize_mode[feature] == 'logicle':
            L = logicleScale.LogicleScale(data)
            L.calculate_y(
                    T = self.normalize_parameters[feature][0],
                    M = self.normalize_parameters[feature][1],
                    A = self.normalize_parameters[feature][2],
                    p = self.normalize_parameters[feature][3],
                    W = self.normalize_parameters[feature][4])
            return L.y
        else:
            raise NotImplementedError('ERROR: mode {0:s} is not implemented'.format(self.normalize_mode[feature]))
    def back_transform(self, feature, data):
        """
        Parameters
        ----------
        feature: str
            Name of the feature
        data: np.ndarray
            One dimensional array with data along that feature
        """
        if isinstance(data,list):
            data = np.array(data)
        if self.normalize_mode[feature] == 'min':
            return data + self.normalize_parameters[feature][0] - self.normalize_parameters[feature][1]
        elif self.normalize_mode[feature] == 'min_max':
            return data * (self.normalize_parameters[feature][1] - self.normalize_parameters[feature][0]) + self.normalize_parameters[feature][0]
        elif self.normalize_mode[feature] == 'mean_std':
            return data * self.normalize_parameters[feature][1] + self.normalize_parameters[feature][0]
        elif self.normalize_mode[feature] == 'arcsinh':
            return self.normalize_parameters[feature][1]*np.sinh(data)
        elif self.normalize_mode[feature] == 'log10':
            dummy = np.power(10.0, data)
            return dummy - self.normalize_parameters[feature][1] + self.normalize_parameters[feature][0]
        elif self.normalize_mode[feature] == 'sqrt':
            dummy = np.power(data,2.0)
            return dummy + self.normalize_parameters[feature]
        elif self.normalize_mode[feature] == 'sigmoid':
            dummy = np.log(np.array(data) / (1.0 - np.array(data)))
            return dummy*self.normalize_parameters[feature][1] + self.normalize_parameters[feature][0]
        elif self.normalize_mode[feature] == 'logicle':
            if not isinstance(data, np.ndarray):
                data = np.array([data,])
            data = data.flatten()
            L = logicleScale.LogicleScale(data)
            original_data = L.calculate_S(
                    y = data,  
                    T = self.normalize_parameters[feature][0],
                    M = self.normalize_parameters[feature][1],
                    A = self.normalize_parameters[feature][2],
                    p = self.normalize_parameters[feature][3],
                    W = self.normalize_parameters[feature][4])
            return original_data
        else:
            raise NotImplementedError('ERROR: mode {0:s} is not implemented'.format(self.normalize_mode[feature]))
    def delete_samples(self, inds, verbose = 0):
        """
        Parameters
        ----------
        inds: np.ndarray
            Indexes of the elements to delete
        """
        if type(inds) == list:
            inds = np.array(inds)
        if inds.dtype == bool:
            inds = np.where(inds)[0]
        if len(inds) and (verbose > 0):
            print('Removing {0:d} samples'.format(len(inds)))
        inds_experiments = []
        for i_experiment, experiment in enumerate(self.experiments):
            inds_experiments.append(self.get_indexes_experiment(inds, i_experiment))
        # Two separate steps are necessary because indexes changes when samples are deleted
        for i_experiment, inds_experiment in enumerate(inds_experiments):
            if len(inds_experiment) and (verbose > 0):
                print('\t{0:d} samples from experiment {1:d}'.format(len(inds_experiment), i_experiment))
            self.experiments[i_experiment].delete_samples(inds_experiment)
    def clean_samples(self, features = None, mode = 'nan'):
        if features is None:
            features = self.get_features()
        if isinstance(features, str):
            features = list([features,])
        for feature in features:
            self.clean_samples_single_feature(feature, mode)
    def clean_samples_single_feature(self, feature, value):
        """
        Remove samples where feature is equal to value

        Parameters
        ----------
        feature: str
            The feature to check
        value:
            The values to remove
            Possible choices:
                'nan' = remove Nan
                'inf' = remove Inf
                '<= x' = remove <= x, x being a float
        """
        data = self.get_data_features([feature])
        if value == 'nan':
            inds_delete = np.where(np.isnan(data))[0]
            mode = 'equal to nan'
        elif value == 'inf':
            inds_delete = np.where(np.isinf(data))[0]
            mode = 'equal to inf'
        elif value[0:2] == '<=':
            x = float(value[2:])
            inds_delete = np.where(data <= x)[0]
            mode = ' <= {0:f}'.format(x)
        elif value[0:2] == '==':
            x = float(value[2:])
            inds_delete = np.where(data == x)[0]
            mode = ' == {0:f}'.format(x)
        elif value[0:2] == '!=':
            x = float(value[2:])
            inds_delete = np.where(data != x)[0]
            mode = ' != {0:f}'.format(x)
        else:
            raise NotImplementedError('ERROR: method {0:s} does not exist for clean_samples'.format(value))
        if len(inds_delete):
            print('Removing {0:d} samples with feature {1:s} {2:s}'.format(len(inds_delete),feature, mode))
        self.delete_samples(inds_delete)
    def remove_norm_zero(self, features):
        """
        Remove samples with normalized data along features lower than zero
        """
        print('Number of samples before negative removal: {0:d}'.format(self.get_n_samples()))
        data = self.get_data_norm_features(features)
        inds_below_zero = np.where(np.any(data <= 0.0, axis = 1))[0]
        self.delete_samples(inds_below_zero)
        print('Number of samples after negative removal: {0:d}'.format(self.get_n_samples()))
    def remove_norm_one(self, features):
        """
        Remove samples with normalized data along features above 1.0
        """
        print('Number of samples before norm-1 removal: {0:d}'.format(self.get_n_samples()))
        data = self.get_data_norm_features(features)
        inds_below_zero = np.where(np.any(data >= 1.0, axis = 1))[0]
        self.delete_samples(inds_below_zero)
        print('Number of samples after norm-1 removal: {0:d}'.format(self.get_n_samples()))
    def remove_outliers(self, features, max_n_std = 5.0, verbose = 0):
        print('Number of samples before outliers removal {0:d}'.format(self.get_n_samples()))
        for feature in features:
            data = self.get_data_features([feature])
            avr = np.mean(data)
            std = np.std(data)
            dist_avr = np.abs(data - avr)
            inds_delete = np.where(dist_avr > max_n_std*std)[0]
            if len(inds_delete) and (verbose > 0):
                print('Removing {0:d} outliers for feature {1:s}'.format(len(inds_delete),feature))
            self.delete_samples(inds_delete)
        print('Number of samples after outliers removal {0:d}'.format(self.get_n_samples()))
    def show(self, features, stride = 0, contour = None, mode = 'experiments', inds_inside = None, pdf = None):
        """
        Scatter plots of samples

        Parameters
        ----------
        features: list of str
            The names of the features
        stride: int
            Sampling period
        countour:
        mode: str / np.ndarray
            Possible values:
            - 'experiments': Use different colors for different experiments
            - 'conditions': Use different colors for different conditions
            - 'density':
            - np.ndarray
        inds_inside: np.ndarray
            Array of boolean values
        """
        if len(features) != 2:
            print('WARNING: scatter plot is possible only with two features')
            return
        features = deepcopy(features)
        data = self.get_data_features([features[0], features[1]])
        data_norm = self.get_data_norm_features([features[0], features[1]])
        #--- Chech if it's necessary to divide the data in inside/outside
        if inds_inside is not None: # divide data in inside and outside
            if not isinstance(inds_inside,np.ndarray):
                raise ValueError('ERROR: wrong format for inds_inside')
            inds_inside = inds_inside.flatten()
            if np.sum(inds_inside) == 0:
                raise ValueError('ERROR: no sample was selected ')
            inds_outside = np.logical_not(inds_inside)
            data_outside = data[inds_outside,:]
            data_norm_outside = data_norm[inds_outside,:]
            data = data[inds_inside,:]
            data_norm = data_norm[inds_inside,:]
        else:
            data_outside = None
            data_norm_outside = None
        #--- Set the sampling period
        if stride == 0:
            stride = max(int(data.shape[0] / 10000),1)
            if data_outside is not None:
                stride_outside = max(int(data_outside.shape[0] / 10000),1)
                data_outside = data_outside[::stride_outside,:]
                data_norm_outside = data_norm_outside[::stride_outside,:]
        else:
            if data_outside is not None:
                data_outside = data_outside[::stride,:]
                data_norm_outside = data_norm_outside[::stride,:]
        data = data[::stride,:]
        data_norm = data_norm[::stride,:]
        #--- Set the method used for coloring
        if isinstance(mode, np.ndarray):
            data_colors = mode.flatten().astype(int)
            if len(data_colors) != self.get_n_samples():
                raise ValueError('ERROR: wrong number of samples in array')
            data_colors = data_colors[::stride]
        elif mode == 'experiments':
            data_colors = np.empty(self.get_n_samples()).astype(int)
            for i_experiment, experiment in enumerate(self.experiments):
                i_start, i_end = self.get_boundaries(i_experiment)
                data_colors[i_start:i_end] = i_experiment
            data_colors = data_colors[::stride]
        elif mode == 'conditions':
            data_colors = np.empty(self.get_n_samples()).astype(int)
            for i_condition, condition in enumerate(self.get_conditions()):
                inds_conditions = self.get_indexes_conditions(condition)
                data_colors[inds_conditions] = i_condition
            data_colors = data_colors[::stride]
        elif mode == 'density':
            data_colors = 'density' # in this case, it is calculated inside scatter_plot
        else:
            raise NotImplementedError('ERROR normalization mode {0:s} not implemented'.format(mode))
        #--- Initialize the figure
        f = plt.figure()
        ax1 = f.add_subplot(111)
        if np.prod(np.isnan(data_norm)):
            if data_outside is not None:
                ax1.plot(data_outside[::stride_outside,0], data_outside[::stride_outside,1], ', ', color  = 'dimgray')
            scatter_plot(ax1,  data, data_colors, contour)
        else:
            if data_outside is not None:
                ax1.plot(data_norm_outside[::stride_outside,0], data_norm_outside[::stride_outside,1], ', ', color  = 'dimgray')
            scatter_plot(ax1,  data_norm, data_colors, contour)
            #--- change labels to actual values / x-axis
            possible_ticks = [-1000,-100,-10,0,10,100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,200000,300000,400000,500000,600000,700000,800000,900000]
            possible_ticklabels = ['-10^3','-10^2','-10^-1','0','10^1','10^2','10^3','','','','','','','','','10^4','','','','','','','','','10^5','','','','','','','','']
            x_norm_min, x_norm_max = ax1.get_xlim()
            x_min, x_max = self.back_transform(features[0], [x_norm_min, x_norm_max])
            x_ticks = []
            x_ticklabels = []
            for ind, x_tick in enumerate(possible_ticks):
                if (x_tick > x_min) and (x_tick < x_max):
                    x_ticks.append(x_tick)
                    x_ticklabels.append(possible_ticklabels[ind])
            x_ticks = np.array(x_ticks)
            x_ticks_norm = self.transform(features[0], x_ticks)
            ax1.set_xticks(x_ticks_norm)
            ax1.set_xticklabels(x_ticklabels)
            #--- change labels to actual values / x-axis
            y_norm_min, y_norm_max = ax1.get_ylim()
            y_min, y_max = self.back_transform(features[0], [y_norm_min, y_norm_max])
            y_ticks = []
            y_ticklabels = []
            for ind, y_tick in enumerate(possible_ticks):
                if (y_tick > y_min) and (y_tick < y_max):
                    y_ticks.append(y_tick)
                    y_ticklabels.append(possible_ticklabels[ind])
            y_ticks = np.array(y_ticks)
            y_ticks_norm = self.transform(features[0], y_ticks)
            ax1.set_yticks(y_ticks_norm)
            ax1.set_yticklabels(y_ticklabels)
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            self.fis = graphics.AxesScaleInteractor(f)
    def show_histogram(self, features = [], nbins = 100, pdf = None):
        """
        Plot 1D histograms

        Parameters
        ----------
        pdf: PdfPages
        features: list
        nbins: int
        """
        if not features:
            features = self.get_features()
        for feature in features:
            data = self.get_data_features([feature,])
            data_norm = self.get_data_norm_features([feature,])
            has_norm = not np.prod(np.isnan(data_norm).flatten())
            f = plt.figure()
            if has_norm:
                ax1 = f.add_subplot(211)
                ax2 = f.add_subplot(212)
            else:
                ax1 = f.add_subplot(111)
            h, e = np.histogram(data, bins = nbins, normed = True)
            b = 0.5*(e[:-1] + e[1:])
            ax1.plot(b,h,'-')
            if has_norm:
                h, e = np.histogram(data_norm, bins = nbins, normed = True)
                b = 0.5*(e[:-1] + e[1:])
                ax2.plot(b,h,'-')
                #--- change labels to actual values / x-axis
                possible_ticks = [-1000,-100,-10,0,10,100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,200000,300000,400000,500000,600000,700000,800000,900000]
                possible_ticklabels = ['-10^3','-10^2','-10^-1','0','10^1','10^2','10^3','','','','','','','','','10^4','','','','','','','','','10^5','','','','','','','','']
                x_norm_min, x_norm_max = ax2.get_xlim()
                x_min, x_max = self.back_transform(feature, [x_norm_min, x_norm_max])
                x_ticks = []
                x_ticklabels = []
                for ind, x_tick in enumerate(possible_ticks):
                    if (x_tick > x_min) and (x_tick < x_max):
                        x_ticks.append(x_tick)
                        x_ticklabels.append(possible_ticklabels[ind])
                x_ticks = np.array(x_ticks)
                x_ticks_norm = self.transform(feature, x_ticks)
                ax2.set_xticks(x_ticks_norm)
                ax2.set_xticklabels(x_ticklabels)
            plt.sca(ax1)
            plt.title(feature)
            if pdf is not None:
                pdf.savefig()
                plt.close()
            else:
                plt.show()
    def show_distributions(self, features, labels, clusters_order = None, pdf = None):
        """
        Parameters
        ----------
        features: list of str
        labels: np.ndarray
        """
        if not isinstance(labels, np.ndarray):
            raise ValueError('ERROR: wrong data format')
        labels = labels.flatten().astype(int)
        if len(labels) != self.get_n_samples():
            raise ValueError('ERROR: wrong number of labels')
        data = self.get_data_norm_features(features)
        features = deepcopy(features)
        dummy, clusters_order = self.plot_distributions(data, labels, features, clusters_order, 'All experiments', pdf)
        if (self.get_n_conditions() > 1): # if more than one conditions exist, make also separate plots and distributions among populations
            percents_conditions = []
            for conditions in self.get_conditions():
                inds_conditions = self.get_indexes_conditions(conditions)
                if pdf is None:
                    percents = np.array([np.sum(labels[inds_conditions] == label) for label in list(set(labels))]).astype(float) # percentage number of elements in each cluster
                    percents /= np.sum(percents)
                else:
                    percents, dummy = self.plot_distributions(data[inds_conditions,:], labels[inds_conditions], features, clusters_order, conditions, pdf) # separate plots, and get back the percents in clusters
                percents_conditions.append(list(percents))
            percents_conditions = np.array(percents_conditions).transpose()
            epsilon = 1e-2*np.min(percents_conditions[percents_conditions > 0])
            percents_conditions[percents_conditions < epsilon] = epsilon
            #--- Occupancies
            f = plt.figure()
            ax = f.add_subplot(121)
            cax = ax.matshow(np.log10(percents_conditions[clusters_order,:]), cmap = 'binary')
            cbar = f.colorbar(cax)
            plt.xticks(range(self.get_n_conditions()),self.get_conditions(),rotation='vertical')
            plt.yticks(range(len(set(labels))),clusters_order)
            plt.ylabel('cluster index')
            cbar.set_label('Occupancies')
            #--- Row Normalized Occupancies
            ax = f.add_subplot(122)
            percents_conditions_norm = (percents_conditions - np.min(percents_conditions, axis = 1).reshape((percents_conditions.shape[0],1))) / (np.max(percents_conditions, axis = 1).reshape((percents_conditions.shape[0],1)) - np.min(percents_conditions, axis = 1).reshape((percents_conditions.shape[0],1)))
            cax = ax.matshow(percents_conditions_norm[clusters_order,:], cmap = 'binary')
            cbar = f.colorbar(cax)
            plt.xticks(range(self.get_n_conditions()),self.get_conditions(),rotation='vertical')
            plt.yticks(range(len(set(labels))),clusters_order)
            plt.ylabel('cluster index')
            cbar.set_label('Row-normalized Occupancies')
            if pdf is not None:
                pdf.savefig()
                plt.close()
        if pdf is None:
            plt.show()
    def plot_distributions(self, data, labels, features, clusters_order, title, pdf = None):
        """
        It's the method actually making the distribution plots

        Parameters
        ----------
        features: list of str
            The names of the features
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        list_labels = list(set(labels))
        n_labels = len(list_labels) 
        n_features = len(features)
        #--- Data table
        percents = np.array([np.sum(labels == label) for label in list_labels]).astype(float) # percentage number of elements in each cluster
        percents /= np.sum(percents)
        data_table = np.zeros((n_features, n_labels))
        std_table = np.zeros((n_features, n_labels))
        text_table = []
        text_row = []
        for i_label, label in enumerate(list_labels):
            text_row.append('{0:6.3f}'.format(percents[i_label]))
        text_table.append(text_row)
        color_table = ['white']
        for i_feature, feature in enumerate(features):
            text_row = []
            color_table.append(settings.colors[i_feature % len(settings.colors)])
            for i_label, label in enumerate(list_labels):
                dummy = data[:,i_feature]
                data_table[i_feature,i_label] = np.mean(dummy[labels == label])
                std_table[i_feature,i_label] = np.std(dummy[labels == label])
                text_row.append('{0:6.3f}'.format(data_table[i_feature,i_label]))
            text_table.append(text_row)
        bar1_width = 1.0*n_features/(n_features + 2)
        bar2_width = 1.0/(n_features + 2)
        bar1_delta = 0.5
        bar2_delta = 1.5*bar2_width
        if pdf is not None:
            f = plt.figure()
            for i_label, label in enumerate(list_labels):
                if percents[i_label] > 0.0:
                    plt.bar(bar1_delta, percents[i_label], bar1_width, color = 'white', linewidth = 1.0, edgecolor = 'black')
                    for i_feature, feature in enumerate(features):
                        scale = ( data_table[i_feature, i_label] - np.min(data_table[i_feature,:]) + 0.1*np.ptp(data_table[i_feature,:]) ) / (1.1*np.ptp(data_table[i_feature,:]))
                        if not np.isnan(scale):
                            plt.bar(bar2_delta, percents[i_label], bar2_width, color = settings.colors[i_feature % len(settings.colors)], linewidth = 0, alpha = scale)
                            bar2_delta += 1.0*bar2_width
                bar2_delta += 2.0*bar2_width 
                bar1_delta += 1
            plt.xlim([0,n_labels])
            table = plt.table(cellText = text_table, rowLabels = ['%',] + features, colLabels = [str(label) for label in list_labels], loc = 'bottom', rowColours=color_table)
            plt.subplots_adjust(left=0.3, bottom=0.3)
            plt.ylabel('Cells in cluster [%]')
            plt.yscale('log')
            plt.xticks([])
            plt.title(title)
            pdf.savefig()
            plt.close()
        average_features = data_table.transpose()
        #--- Dendogram
        if clusters_order is None:
            f = plt.figure()
            Z = linkage(average_features)
            R = dendrogram(Z)
            clusters_order = R['leaves']
            if pdf is not None:
                pdf.savefig()
                plt.close()
        #--- Heatmap
        f = plt.figure()
        ax = f.add_subplot(121)
        #cax = ax.matshow(np.log10(average_features[clusters_order,:]), cmap = 'RdYlBu_r')
        cax = ax.matshow(average_features[clusters_order,:], cmap = 'RdYlBu_r')
        cbar = f.colorbar(cax)
        #possible_ticks = [-1000,-100,-10,-1,0,1,10,100,1000,10000,100000]
        #possible_ticklabels = ['-10^3','-10^2','-10^-1','-10^0','0','10^0','10^1','10^2','10^3','10^4','10^5']
        possible_ticks = [-1000,-100,-10,0,10,100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,100000,200000,300000,400000,500000,600000,700000,800000,900000]
        possible_ticklabels = ['-10^3','-10^2','-10^-1','0','10^1','10^2','10^3','','','','','','','','','10^4','','','','','','','','','10^5','','','','','','','','']
        c_norm_min, c_norm_max = cax.get_clim()
        c_min_max = self.back_transform(features[0], [c_norm_min, c_norm_max])
        c_min = c_min_max[0]
        c_max = c_min_max[1]
        c_ticks = []
        c_ticklabels = []
        for ind, c_tick in enumerate(possible_ticks):
            if (c_tick > c_min) and (c_tick < c_max):
                c_ticks.append(c_tick)
                c_ticklabels.append(possible_ticklabels[ind])
        c_ticks = np.array(c_ticks)
        c_ticks_norm = self.transform(features[0], c_ticks)
        cbar.set_ticks(c_ticks_norm)
        cbar.set_ticklabels(c_ticklabels)
        plt.xticks(range(len(features)),features,rotation='vertical')
        plt.yticks(range(len(set(labels))),clusters_order)
        plt.ylabel('cluster index')
        cbar.set_label('Heatmap - ' + title)
        #--- Normalized Heatmap
        average_features_norm = (average_features - np.min(average_features, axis = 0)) / (np.max(average_features, axis = 0) - np.min(average_features, axis = 0))
        ax = f.add_subplot(122)
        cax = ax.matshow(average_features_norm[clusters_order,:], cmap = 'RdYlBu_r', vmin = 0, vmax = 1)
        cbar = f.colorbar(cax)
        plt.xticks(range(len(features)),features,rotation='vertical')
        plt.yticks(range(len(set(labels))),clusters_order)
        plt.ylabel('cluster index')
        cbar.set_label('Normalized heatmap - ' + title)
        if pdf is not None:
            pdf.savefig()
            plt.close()
        return percents, clusters_order
    def write(self, file_name):
        with open(file_name,'wb') as fout:
            pickle.dump(self, fout)
    def __str__(self):
        if self.get_n_experiments() == 0:
            return 'No experimental data'
        output  = 'Number of experiments: {0:d}\n'.format(self.get_n_experiments())
        output += 'Number of sample: {0:d}\n'.format(self.get_n_samples())
        output += 'Number of common features: {0:d}\n'.format(len(self.get_features()))
        output += 'Common features: {0:s}\n'.format(','.join(self.get_features()))
        for conditions in self.get_conditions():
            output += 'Condition {0:s}\n'.format(conditions)
            output += '\tNumber of experiments: {0:d}\n'.format(self.get_n_experiments(conditions))
            output += '\tNumber of sample: {0:d}\n'.format(self.get_n_samples(conditions))
        for i_feature, feature in enumerate(self.get_features()):
            output += 'Feature = {0:s}\n'.format(feature)
            output += '\tmin = {0:f} | {1:f}\n'.format(self.get_min_feature(feature),self.get_min_norm_feature(feature))
            output += '\tmax = {0:f} | {1:f}\n'.format(self.get_max_feature(feature),self.get_max_norm_feature(feature))
            output += '\tmean = {0:f} | {1:f}\n'.format(self.get_mean_feature(feature),self.get_mean_norm_feature(feature))
            output += '\tstd = {0:f} | {1:f}\n'.format(self.get_std_feature(feature),self.get_std_norm_feature(feature))
        return output[:-1]
 
class Experiment(object):
    """
    Single FCS experiment

    Attributes
    ----------
    sample: object of <class 'FlowCytometryTools.core.containers.FCMeasurement'>
    data: np.array
        Raw data
    data_norm: np.array
        Normalized data
    features: list
        Name of the features
    features_synonym: list
        Alternative names for the features
    features_normalized: list
        Name of the features that were normalized
    fis: list
        Variables used to store the interactive figures
    """
    def __init__(self, file_name, mode = 'all', id_sample = 0):
        """
        Parameters
        ----------
        file_name: str
            Name of FCS file
        mode: str
            How to read the data:
                'all' = read everything
                int = choose INT elements at random, this is usefull for quick tests of the code
        id_sample: int
        """
        #--- Initialize data
        self.sample = FCMeasurement(ID = id_sample, datafile = file_name)
        if mode == 'all':
            self.data = np.copy(self.sample.data.values[:,:])
        elif isinstance(mode,int):
            inds_samples = np.random.choice(range(self.sample.data.values.shape[0]), size = mode, replace = False)
            self.data = np.copy(self.sample.data.values[inds_samples,:])
        else:
            raise ValueError('ERROR: wrong reading mode {0}'.format(mode))
        print('Read {0:d} samples from {1:s}'.format(self.get_n_samples(), file_name))
        #--- Initialized other attributes
        self.data_norm = np.empty((self.get_n_samples(),0))
        self.features = list(self.sample.data.columns)
        self.features_synonym = deepcopy(self.features)
        self.features_norm = []
        for key, value in self.sample.meta.items():
            if len(key) > 2:
                if (key[0:2] == '$P') and (key[-1] == 'S'):
                    index = int(key[2:-1])-1
                    self.features_synonym[index] = value
        self.fis = []
    def add_feature(self, feature, data):
        """
        Parameters
        ----------
        feature: str
            The name of the feature to add
        data: np.ndarray / list
            The values to add
        """
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise ValueError('ERROR: wrong data format')
        if len(data) != self.get_n_samples():
            raise ValueError('ERROR: wrong number of values for feature {0:s}'.format(feature))
        data = data.reshape(-1,1)
        if feature not in self.features:
            self.data = np.hstack((self.data, data))
            self.features.append(feature)
            self.features_synonym.append(feature)
        else:
            print('WARNING: {0:s} already exist, overwritting'.format(feature))
            self.data[:,self.get_index_features([feature,])] = data
    def get_n_samples(self):
        """
        Return
        ------
        int
            Number of samples
        """
        return self.data.shape[0]
    def get_data(self):
        """
        Return
        ------
        np.array
            Data in the experiment
        """
        return self.data
    def get_data_norm(self):
        """
        Return
        ------
        np.array
            Normalized data in the experiment
        """
        return self.data_norm
    def get_data_features(self, features):
        if not isinstance(features, list):
            if not isinstance(features, str):
                raise ValueError('ERROR: wrong format for feature')
            features = [features,]
        return self.data[:,self.get_index_features(features)]
    def get_data_norm_features(self, features):
        if not isinstance(features, list):
            if not isinstance(features, str):
                raise ValueError('ERROR: wrong format for feature')
            features = [features,]
        return self.data_norm[:,self.get_index_norm_features(features)]
    def get_index_features(self, features):
        ind_features = []
        for feature in features:
            if feature in self.get_features():
                ind_features.append(self.get_features().index(feature))
            elif feature in self.get_features_synonym():
                ind_features.append(self.get_features_synonym().index(feature))
            else:
                raise ValueError('Feature {0:s} does not exist'.format(feature))
        return ind_features
    def get_index_norm_features(self, features):
        ind_features = []
        for feature in features:
            if feature in self.get_norm_features():
                ind_features.append(self.get_norm_features().index(feature))
            else:
                raise ValueError('Feature {0:s} was not normalized'.format(feature))
        return ind_features
    def get_features(self):
        return self.features
    def get_features_synonym(self):
        return self.features_synonym
    def get_norm_features(self):
        return self.features_norm
    def get_compensation(self):
        try:
            spill_str = self.sample.meta['SPILL']
        except:
            raise ValueError('ERROR: SPILL keyword missing from metadata')
        spill = spill_str.split(',')
        n_features = int(spill[0])
        name_features = spill[1:n_features+1]
        ind_features = self.get_index_features(name_features)
        compensate_matrix= np.array([float(x) for x in spill[n_features+1:]]).reshape((n_features, n_features))
        return ind_features, compensate_matrix
    def has_norm(self, features):
        """
        Return True if normalized data were defined for this experiment

        Parameters
        ----------
        feature: list
            If defined, it checks if the features in the list were normalized
            Otherwise, the return value is True if any feature was normalized
        """
        if not isinstance(features, list):
            if not isinstance(features, str):
                raise ValueError('ERROR: wrong format for feature')
            features = [features,]
        return all([feature in self.get_norm_features() for feature in features])
    def compensate(self):
        ind_features, compensate_matrix = self.get_compensation()
        self.data[:,ind_features] = np.dot(self.data[:,ind_features],np.linalg.inv(compensate_matrix))
    def normalize(self, feature, transform_function):
        """
        Parameters
        ---------
        feature: str
        transform_function: a function with parameters the feature and the data and that returns the normalized data
        """
        ind_feature = self.get_index_features([feature])
        if self.has_norm(feature):
            ind_norm_feature = self.get_index_norm_features([feature])[0]
            self.data_norm[:,ind_norm_feature] = transform_function(feature, self.data[:,ind_feature]).flatten()
        else:
            self.data_norm = np.hstack((self.data_norm, transform_function(feature, self.data[:,ind_feature])))
            self.features_norm.append(feature)
    def delete_samples(self, inds):
        """
        Remove samples
        """
        self.data = np.delete(self.data, inds, axis = 0)
        self.data_norm = np.delete(self.data_norm, inds, axis = 0)
    def show(self, features,  stride = 0, title = None, pdf = None):
        """
        features: list
        stride: int
        title: str
        """
        if len(features) != 2:
            print('WARNING: show works only for 2 features')
            return
        if stride == 0:
            stride = max(int(self.get_n_samples() / 10000),1)
        data = self.get_data_features([features[0], features[1]])
        if self.has_norm(features):
            data_norm = self.get_data_norm_features([features[0], features[1]])
        f = plt.figure()
        if self.has_norm(features): 
            ax1 = f.add_subplot(211)
            ax2 = f.add_subplot(212)
        else:
            ax1 = f.add_subplot(111)
        plt.sca(ax1)
        ax1.plot(data[::stride,0], data[::stride,1],',k')
        plt.ylabel(features[1])
        if self.has_norm(features): 
            plt.sca(ax2)
            ax2.plot(data_norm[::stride,0], data_norm[::stride,1],',k')
            plt.xlabel(features[0])
            plt.ylabel(features[1])
        else:
            plt.xlabel(features[0])
        if title is not None:
            plt.sca(ax1)
            plt.title(title)
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            self.fis.append(graphics.AxesScaleInteractor(f))
    def __str__(self):
        output  = 'Number of samples: {0:d}\n'.format(self.get_n_samples())
        output += 'Features\n'
        features = self.get_features()
        features_syn = self.get_features_synonym()
        for index, feature in enumerate(features):
            if feature == features_syn[index]:
                output += '\t{0:d}\t{1:s}\n'.format(index, feature)
            else:
                output += '\t{0:d}\t{1:s}\t{2:s}\n'.format(index, feature, features_syn[index])
        return output[:-1]

if __name__ == '__main__':
    print('---------------')
    print('Testing data.py')
    print('---------------')

    pdf = PdfPages('./test.pdf')
    C = Collection() # Create a collection of experiments
    E1 = Experiment(file_name = '../examples/data/flowc/levine_13dim.fcs', mode = 50000) # Read data for 1st experiment
    C.add_experiment(E1, condition = 'random_set_1') # Add data to collection
    E2 = Experiment(file_name = '../examples/data/flowc/levine_13dim.fcs', mode = 50000) # Read data for 2nd experiment
    C.add_experiment(E2, condition = 'random_set_2') # Add data to collection
    features = ['CD45','CD20','CD38'] # Choose some features
    C.clean_samples() # Remove nan from all features
    #C.compensate() # Compensate data
    C.normalize(features = features, mode = 'logicle') # Normalize data
    E1.show(features[1:3], pdf = pdf) # Show a single experiment
    C.show(features[1:3], stride = 0, mode = 'experiments', pdf = pdf) # Show scattered data, colored according to experiment index
    C.show(features[1:3], stride = 0, mode = 'conditions', pdf = pdf) # Show scattered data, colored according to condition index
    C.show(features[1:3], stride = 0, mode = 'density', pdf = pdf) # Show scattered data, colored according to density
    C.show(features[1:3], stride = 0, mode = C.get_data_features(['label']), pdf = pdf) # Show scattered data, colored using an np.ndarray
    C.show(features[1:3], stride = 0, mode = 'density', inds_inside = C.get_data_features(['label']) == 1, pdf = pdf) # Show scattered data, colored according to density
    C.show_histogram(pdf = pdf) # Show histograms for all the experiments in the collection
    C.show_distributions(features, C.get_data_features(['label']), pdf = pdf) # Show cluster distribution
    pdf.close()
