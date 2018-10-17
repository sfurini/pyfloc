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

#plt.switch_backend('agg')

np.set_printoptions(linewidth = np.inf)
print = functools.partial(print, flush=True)

class Collection(object):
    """
    Set of FCS experiments

    Attributes
    ----------
    conditions: list of str
    experiments: list of Experiment
    normalize_parameters: dict
        key: str
            Feature names
        values: list of float
    labels: np.ndarray
        Array of int, with the labels
    fis: list
        Variables used to keep the interactive figures
    """
    def __init__(self):
        self.conditions = []
        self.experiments = []
        self.normalize_parameters = {}
        self_transform_mode = ''
        self.labels = np.empty(0)
        self.fis = None
    def add_experiment(self, experiment, condition, labels = None):
        """
	Parameters
	----------
        experiment: Experiment
        condition: str
        labels: np.ndarray
            Same len as number of samples
            nan values are converted to -1
        """
        self.conditions.append(condition)
        self.experiments.append(experiment)
        if labels is None:
            self.labels = np.hstack((self.labels, -1*np.ones(experiment.get_n_samples()))).astype(int)
        else:
            if len(labels) != experiment.get_n_samples():
                raise ValueError('ERROR: number of labels different from number of samples')
            labels[np.isnan(labels)] = -1
            self.labels = np.hstack((self.labels, labels.flatten())).astype(int)
    def get_conditions(self):
        set_conditions = set(self.conditions)
        set_conditions = ['9','1','2','3']
        # BEGIN: SOLUZIONE TEMPORANEA
        #list_conditions = []
        #for organ in ['BM', 'ILN', 'SPL']:
        #    for day in ['pre', 'd3', 'd7']:
        #        for condition in set_conditions:
        #            if (organ in condition) and (day in condition):
        #                list_conditions.append(condition)
        # END: SOLUZIONE TEMPORANEA
        return list(set_conditions)
    def get_n_experiments(self, conditions = None):
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
            Data in the experiment
        """
        data = np.empty((0,len(features)))
        for experiment in self.experiments:
            data_experiment = experiment.get_data_features(features)
            data = np.vstack((data, data_experiment))
        return data
    def get_min_feature(self, feature):
        return np.nanmin(self.get_data_features([feature]))
    def get_max_feature(self, feature):
        return np.nanmax(self.get_data_features([feature]))
    def get_mean_feature(self, feature):
        return np.nanmean(self.get_data_features([feature]))
    def get_std_feature(self, feature):
        return np.nanstd(self.get_data_features([feature]))
    def get_data_norm_features(self, features):
        """
        Parameters
        ----------
        features: list
            The features to extract

        Return
        ------
        np.array
            Normalized data in the experiment
        """
        data_norm = np.empty((0,len(features)))
        for experiment in self.experiments:
            data_norm_experiment = experiment.get_data_norm_features(features)
            data_norm = np.vstack((data_norm, data_norm_experiment))
        return data_norm
    def get_min_norm_feature(self, feature):
        return np.min(self.get_data_norm_features([feature]))
    def get_max_norm_feature(self, feature):
        return np.max(self.get_data_norm_features([feature]))
    def get_mean_norm_feature(self, feature):
        return np.mean(self.get_data_norm_features([feature]))
    def get_std_norm_feature(self, feature):
        return np.std(self.get_data_norm_features([feature]))
    def get_boundaries(self, i_experiment):
        ind_start = 0
        ind_end = self.experiments[0].get_n_samples()
        for i in range(1,i_experiment+1):
            ind_start += self.experiments[i-1].get_n_samples()
            ind_end += self.experiments[i].get_n_samples()
        return ind_start, ind_end
    def get_indexes_conditions(self, conditions):
        """
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
                min_feature = self.get_min_feature(feature)
                self.normalize_parameters[feature] = min_feature
            elif kwargs['mode'] == 'min_max':
                min_feature = self.get_min_feature(feature)
                max_feature = self.get_max_feature(feature)
                self.normalize_parameters[feature] = [min_feature, max_feature]
            elif kwargs['mode'] == 'mean_std':
                mean_feature = self.get_mean_feature(feature)
                std_feature = self.get_std_feature(feature)
                self.normalize_parameters[feature] = [mean_feature, std_feature]
            elif kwargs['mode'] == 'arcsinh':
                self.normalize_parameters[feature] = [kwargs.get('bias',0.0), kwargs.get('factor',5.0)]
                if self.normalize_parameters[feature][0] == 'min':
                    min_feature = self.get_min_feature(feature)
                    self.normalize_parameters[feature][0] = min_feature
            elif kwargs['mode'] == 'logicle':
                data_feature = self.get_data_features([feature])
                L = logicleScale.LogicleScale(data_feature)
                L.calculate_T_M_A_r()
                L.calculate_p_W()
                self.normalize_parameters[feature] = [L.T, L.M, L.A, L.p, L.W] 
            for i_experiment, experiment in enumerate(self.experiments):
                if (verbose > 0):
                    print('Running normalization for feature {0:s} in experiment {1:d} with mode {2:s}'.format(feature, i_experiment,kwargs.get('mode','min')))
                experiment.normalize(feature, self.normalize_parameters, **kwargs)
        self.transform_mode = kwargs['mode']    
    def transform(self, feature, data):
        """
        Parameters
        ----------
        feature: str
            Name of the feature
        data: np.ndarray
            One dimensional array with data along that feature
        """
        if self.transform_mode == 'arcsinh':
            return np.arcsinh((data - self.normalize_parameters[feature][0])/self.normalize_parameters[feature][1])
        elif self.transform_mode == 'logicle':
            L = logicleScale.LogicleScale(data)
            L.calculate_y(
                    T = self.normalize_parameters[feature][0],
                    M = self.normalize_parameters[feature][1],
                    A = self.normalize_parameters[feature][2],
                    p = self.normalize_parameters[feature][3],
                    W = self.normalize_parameters[feature][4])
            return L.y
        else:
            raise NotImplementedError('ERROR: mode {0:s} is not implemented'.format(self.transform_mode))
    def back_transform(self, feature, data):
        """
        Parameters
        ----------
        feature: str
            Name of the feature
        data: np.ndarray
            One dimensional array with data along that feature
        """
        if self.transform_mode == 'arcsinh':
            return self.normalize_parameters[feature][1]*np.sinh(data)
        elif self.transform_mode == 'logicle':
            L = logicleScale.LogicleScale(data)
            original_data = L.calculate_S(
                    y = self.get_data_norm_features([feature]),  
                    T = self.normalize_parameters[feature][0],
                    M = self.normalize_parameters[feature][1],
                    A = self.normalize_parameters[feature][2],
                    p = self.normalize_parameters[feature][3],
                    W = self.normalize_parameters[feature][4])
            return [np.min(original_data), np.max(original_data)]
        else:
            raise NotImplementedError('ERROR: mode {0:s} is not implemented'.format(self.transform_mode))
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
        # Remove labels of deleted samples
        if len(inds) > 0:
            self.labels[inds] = -123456789 # just a random number to mark them for delete
            self.labels = self.labels[self.labels != -123456789]
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
    def show_scatter(self, features, features_synonym = {}, stride = 0, contour = None, mode = 'experiments', inds_inside = None, pdf = None):
        """
        Scatter plots of samples. It works only with 2 features
        The differences between this method and self.show are:
        - self.show only plot dots for all the experiments one by one
        - this method plots all the experiments together (plus separate plots if using pdf)

        Parameters
        ----------
        features: list of str
            The names of the features
        features_synonym:   dict
            key: features
            values: alternative names
        stride: int
            Sampling period
        countout:
        """
        if len(features) != 2:
            print('WARNING: scatter plot is possible only with two features')
            return
        features = deepcopy(features)
        data = self.get_data_features([features[0], features[1]])
        data_norm = self.get_data_norm_features([features[0], features[1]])
        if inds_inside is not None: # divide data in inside and outside
            inds_outside = np.logical_not(inds_inside)
            data_outside = data[inds_outside,:]
            data_norm_outside = data_norm[inds_outside,:]
            data = data[inds_inside,:]
            data_norm = data_norm[inds_inside,:]
        else:
            data_outside = None
            data_norm_outside = None
        for i_feature, feature in enumerate(features):
            features[i_feature] = features_synonym.get(feature, feature)
        if mode == 'experiments':
            data_colors = np.empty(data.shape[0]).astype(int)
            for i_experiment, experiment in enumerate(self.experiments):
                i_start, i_end = self.get_boundaries(i_experiment)
                data_colors[i_start:i_end] = i_experiment
        elif mode == 'conditions':
            data_colors = np.empty(data.shape[0]).astype(int)
            for i_condition, condition in enumerate(self.get_conditions()):
                inds_conditions = self.get_indexes_conditions(condition)
                data_colors[inds_conditions] = i_condition
        elif mode == 'labels':
            data_colors = self.labels
        elif mode == 'density':
            data_colors = 'density' # in this case, it is calculated inside single_scatter_plot
        else:
            raise NotImplementedError('ERROR normalization mode {0:s} not implemented'.format(mode))
        self.single_scatter_plot(data, data_norm, data_colors, features, stride, contour, '', pdf, data_outside, data_norm_outside)
        #if (pdf is not None) and (len(self.get_conditions()) > 1): # if more than one conditions exist, make also separate plots (but only if we're plotting on PDF)
        #    for conditions in self.get_conditions():
        #        inds_conditions = self.get_indexes_conditions(conditions)
        #        try:
        #            self.single_scatter_plot(data[inds_conditions,:], data_norm[inds_conditions,:], data_colors[inds_conditions], features, stride, contour, conditions, pdf)
        #        except:
        #            self.single_scatter_plot(data[inds_conditions,:], data_norm[inds_conditions,:], data_colors, features, stride, contour, conditions, pdf)
    def single_scatter_plot(self, data, data_norm, data_colors, features, stride, contour, title, pdf, data_outside = None, data_norm_outside = None):
        """
        It's the method actually making the scatter plot
        """
        if stride == 0:
            stride = max(int(data.shape[0] / 20000),1)
            if data_outside is not None:
                stride_outside = max(int(data_outside.shape[0] / 10000),1)
        else:
            if data_outside is not None:
                stride_outside = stride
        data = data[::stride,:]
        data_norm = data_norm[::stride,:]
        has_norm = not np.prod(np.isnan(data_norm))
        f = plt.figure()
        ax1 = f.add_subplot(111)
        #if has_norm:
        #    ax1 = f.add_subplot(311)
        #    ax2 = f.add_subplot(312)
        #else:
        #    ax1 = f.add_subplot(211)
        #    ax2 = f.add_subplot(212)
        #if data_outside is not None:
        #    ax1.plot(data_outside[::stride_outside,0], data_outside[::stride_outside,1], ', ', color  = 'dimgray')
        #    ax2.plot(data_outside[::stride_outside,0], data_outside[::stride_outside,1], ', ', color  = 'dimgray')
        #self.single_scatter_panel(ax1, data, data_colors, contour, stride)
        #self.single_scatter_panel(ax2, data, data_colors, contour, stride)
        #plt.sca(ax1)
        #plt.ylabel(features[1])
        #plt.sca(ax2)
        #plt.xlim([data[data[:,0] > 0,0].min(), data[:,0].max()])
        #plt.ylim([data[data[:,1] > 0,1].min(), data[:,1].max()])
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.ylabel(features[1])
        #if not isinstance(data_colors,str):
        #    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) 
        if not has_norm:
            if data_outside is not None:
                ax1.plot(data_outside[::stride_outside,0], data_outside[::stride_outside,1], ', ', color  = 'dimgray')
            self.single_scatter_panel(ax1,  data, data_colors, contour, stride)
        else:
            if data_outside is not None:
                ax1.plot(data_norm_outside[::stride_outside,0], data_norm_outside[::stride_outside,1], ', ', color  = 'dimgray')
            self.single_scatter_panel(ax1,  data_norm, data_colors, contour, stride)
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
        plt.ylabel(features[1])
        plt.xlabel(features[0])
        plt.title(title)
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            self.fis = graphics.AxesScaleInteractor(f)
    def single_scatter_panel(self, ax, data, data_colors, contour, stride):
        if isinstance(data_colors,str):
            if data_colors == 'density':
                from scipy.stats import binned_statistic_2d
                H, xe, ye, ix_iy = binned_statistic_2d(data[:,0], data[:,1], None, statistic = 'count', bins = 100, range = [[data[:,0].min(), data[:,0].max()],[data[:,1].min(), data[:,1].max()]], expand_binnumbers = True)
                ix_iy -= 1
                data_colors = H[ix_iy[0,:], ix_iy[1,:]]
                data_colors = np.log10(data_colors)
                ax.scatter(data[:,0], data[:,1], marker = ',', s = 1.0, c = data_colors, cmap = 'inferno')
                #plt.xlim([data[:,0].min(), data[:,0].max()])
                #plt.ylim([data[:,1].min(), data[:,1].max()])
        else:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.plot(data[:,0], data[:,1],',k',markersize = 1.0, label = '')
            data_colors = data_colors[::stride]
            for data_color in set(data_colors):
                inds_data_color = (data_colors == data_color)
                ax.plot(data[inds_data_color,0], data[inds_data_color,1], '.', markersize = 2.0, color = settings.colors[data_color%len(settings.colors)], label = str(data_color))
        if contour is not None:
            ax.plot(contour[0], contour[1], '--r')
        #from sklearn import mixture
        #for data_color in set(data_colors):
        #    inds_data_color = (data_colors == data_color)
        #    if len(data[inds_data_color,0]) > 100: 
        #        H, xe, ye = np.histogram2d(np.log10(data[inds_data_color,0]), np.log10(data[inds_data_color,1]), bins = 200, range = [[1,4.5], [1,4]])
        #        xb = 0.5*(xe[:-1]+xe[1:])
        #        yb = 0.5*(ye[:-1]+ye[1:])
        #        X, Y = np.meshgrid(xb,yb)
        #        X = np.transpose(X)
        #        Y = np.transpose(Y)
        #        clf = mixture.GaussianMixture(n_components = 50, covariance_type = 'full')
        #        data_log10 = np.log10(data[inds_data_color,:])
        #        inds_del = np.where(np.sum(np.logical_not(np.isfinite(data_log10)), axis = 1))[0]
        #        data_log10 = np.delete(data_log10, inds_del, axis = 0)
        #        clf.fit(data_log10)
        #        XY = np.array([X.ravel(), Y.ravel()]).T
        #        Z = clf.score_samples(XY)
        #        Z = Z.reshape(X.shape)
        #        Z -= np.min(Z) 
        #        Z /= np.sum(Z)
        #        Zlog = np.log10(Z)
        #        Zlog[np.logical_not(np.isfinite(Zlog))] = np.nan
        #        f = plt.figure()
        #        ax = f.add_subplot(111)
        #        box = ax.get_position()
        #        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #        ax.plot(data[:,0], data[:,1],',k',markersize = 1.0, label = 'all samples')
        #        ax.plot(data[inds_data_color,0], data[inds_data_color,1], '.', markersize = 2.0, color = settings.colors[data_color%len(settings.colors)], label = str(data_color))
        #        plt.title(str(data_color))
        #        plt.xlabel(features[0])
        #        plt.ylabel(features[1])
        #        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        #        pdf.savefig()
        #        plt.xscale('log')
        #        plt.yscale('log')
        #        pdf.savefig()
        #        plt.close()
        #        f = plt.figure()
        #        ax = f.add_subplot(111)
        #        ax.pcolormesh(X, Y, np.log10(H), cmap = plt.get_cmap('hot'))
        #        ax.contour(X, Y, Zlog, levels = np.linspace(np.nanmin(Zlog), np.nanmax(Zlog), 100), cmap = plt.get_cmap('winter'))
        #        pdf.savefig()
        #        plt.close()
    def show_distributions(self, features, features_synonym = {}, clusters_order = None, pdf = None):
        """
        """
        if len(set(self.labels)) == 1:
            print('WARNING: no distribution to plot if labels are not defined')
            return
        data = self.get_data_norm_features(features)
        features = deepcopy(features)
        for i_feature, feature in enumerate(features):
            features[i_feature] = features_synonym.get(feature, feature)
        labels = self.labels
        dummy, clusters_order = self.plot_distributions(data, labels, features, clusters_order, 'All experiments', pdf)
        if (len(self.get_conditions()) > 1): # if more than one conditions exist, make also separate plots and distributions among populations
            percents_conditions = []
            for conditions in self.get_conditions():
                inds_conditions = self.get_indexes_conditions(conditions)
                if pdf is None:
                    percents = np.array([np.sum(labels[inds_conditions] == label) for label in list(set(self.labels))]).astype(float) # percentage number of elements in each cluster
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
            plt.xticks(range(len(self.get_conditions())),self.get_conditions(),rotation='vertical')
            plt.yticks(range(len(set(labels))),clusters_order)
            plt.ylabel('cluster index')
            cbar.set_label('Occupancies')
            #--- Row Normalized Occupancies
            ax = f.add_subplot(122)
            percents_conditions_norm = (percents_conditions - np.min(percents_conditions, axis = 1).reshape((percents_conditions.shape[0],1))) / (np.max(percents_conditions, axis = 1).reshape((percents_conditions.shape[0],1)) - np.min(percents_conditions, axis = 1).reshape((percents_conditions.shape[0],1)))
            cax = ax.matshow(percents_conditions_norm[clusters_order,:], cmap = 'binary')
            cbar = f.colorbar(cax)
            plt.xticks(range(len(self.get_conditions())),self.get_conditions(),rotation='vertical')
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
        list_labels = list(set(self.labels))
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
        c_min, c_max = self.back_transform(features[0], [c_norm_min, c_norm_max])
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
    def show(self, feature_0, feature_1, stride = 0, pdf = None):
        """
        """
        for i_experiment, experiment in enumerate(self.experiments):
            condition = self.conditions[i_experiment]
            experiment.show(feature_0, feature_1, stride, condition, pdf)
    def write(self, file_name):
        with open(file_name,'wb') as fout:
            pickle.dump(self, fout)
    def __str__(self):
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
    sample: <class 'FlowCytometryTools.core.containers.FCMeasurement'>
    data: np.array
        Raw data
    data_norm: np.array
        Normalized data
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
        self.sample = FCMeasurement(ID = id_sample, datafile = file_name)
        if mode == 'all':
            self.data = np.copy(self.sample.data.values[:,:])
        elif isinstance(mode,int):
            inds_samples = np.random.choice(range(self.sample.data.values.shape[0]), size = mode, replace = False)
            self.data = np.copy(self.sample.data.values[inds_samples,:])
        else:
            raise ValueError('ERROR: wrong reading mode {0}'.format(mode))
        self.data_norm = np.nan*np.ones(np.shape(self.data))
        self.fis = []
        print('Read {0:d} samples from {1:s}'.format(self.get_n_samples(), file_name))
    def randomize(self):
        """
        """
        pass
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
            features = [features,]
        return self.data[:,self.get_index_features(features)]
    def get_data_norm_features(self, features):
        return self.data_norm[:,self.get_index_features(features)]
    def get_index_features(self, features):
        ind_features = []
        for feature in features:
            if feature in self.get_features():
                ind_features.append(self.get_features().index(feature))
            elif feature in self.get_features_synonym():
                ind_features.append(self.get_features_synonym().index(feature))
            else:
                print(type(features))
                print(features)
                print(feature)
                raise ValueError('Feature {0:s} does not exist'.format(feature))
        return ind_features
    def get_features(self):
        """
        Return
        ------
        list
            Name of the features
        """
        return list(self.sample.data.columns)
    def get_features_synonym(self):
        features = deepcopy(self.get_features())
        for key, value in self.sample.meta.items():
            if len(key) > 2:
                if (key[0:2] == '$P') and (key[-1] == 'S'):
                    index = int(key[2:-1])-1
                    #print(key,' = ',value,' = ',index,' = ',features[index])
                    features[index] = value
        return features
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
    def has_norm(self):
        """
        Return True if normalized data were defined for this experiment
        """
        return not np.prod(np.isnan(self.data_norm))
    def compensate(self):
        ind_features, compensate_matrix = self.get_compensation()
        self.data[:,ind_features] = np.dot(self.data[:,ind_features],np.linalg.inv(compensate_matrix))
    def normalize(self, feature, norm_parameters, **kwargs):
        """
        Parameters
        ---------
        feature:    str
        mode: str
            See code for definition of the implemented normalization mode
        normalize_bias: float
        normalize_factor: float
            These are parameters that can be used to tune the normalization functions
        """
        ind_features = self.get_index_features([feature])
        if kwargs['mode'] == 'min':
            self.data_norm[:,ind_features] = (self.data[:,ind_features] - norm_parameters[feature])
        elif kwargs['mode'] == 'min_max':
            self.data_norm[:,ind_features] = (self.data[:,ind_features] - norm_parameters[feature][0]) / (norm_parameters[feature][1] - norm_parameters[feature][0])
        elif kwargs['mode'] == 'mean_std':
            self.data_norm[:,ind_features] = (self.data[:,ind_features] - norm_parameters[feature][0]) / norm_parameters[feature][1]
        elif kwargs['mode'] == 'arcsinh':
            dummy = (self.data[:,ind_features] - norm_parameters[feature][0])
            self.data_norm[:,ind_features] = np.arcsinh( dummy /  norm_parameters[feature][1])
        elif kwargs['mode'] == 'logicle':
            L = logicleScale.LogicleScale(self.data[:,ind_features])
            self.data_norm[:,ind_features] = L.calculate_y(
                    T = norm_parameters[feature][0], 
                    M = norm_parameters[feature][1],
                    A = norm_parameters[feature][2], 
                    p = norm_parameters[feature][3],
                    W = norm_parameters[feature][4])
        else:
            raise NotImplementedError('ERROR normalization mode {0:s} not implemented'.format(kwargs['mode']))
    def delete_samples(self, inds):
        self.data = np.delete(self.data, inds, axis = 0)
        self.data_norm = np.delete(self.data_norm, inds, axis = 0)
    def show(self, feature_0, feature_1,  stride = 0, title = None, pdf = None):
        """
        feature_0: str
        feature_1: str
        stride: int
        title: str
        """
        if stride == 0:
            stride = max(int(self.get_n_samples() / 10000),1)
        data = self.get_data_features([feature_0, feature_1])
        if self.has_norm():
            data_norm = self.get_data_norm_features([feature_0, feature_1])
        f = plt.figure()
        if self.has_norm():
            ax1 = f.add_subplot(211)
            ax2 = f.add_subplot(212)
        else:
            ax1 = f.add_subplot(111)
        plt.sca(ax1)
        ax1.plot(data[::stride,0], data[::stride,1],',k')
        plt.ylabel(feature_1)
        if self.has_norm():
            plt.sca(ax2)
            ax2.plot(data_norm[::stride,0], data_norm[::stride,1],',k')
            plt.xlabel(feature_0)
            plt.ylabel(feature_1)
        else:
            plt.xlabel(feature_0)
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

    # Create a collection of experiments
    C = Collection()
    # Read data for 1st experiment
    E1 = Experiment(file_name = '../examples/data/flowc/blood.fcs', mode = 50000)
    print(E1.get_features_synonym())
    # Add data to collection
    C.add_experiment(E1, condition = 'random_set_1') #, labels = E1.get_data_features(['label']))
    # Read data for 2nd experiment
    E2 = Experiment(file_name = '../examples/data/flowc/blood.fcs', mode = 50000)
    # Add data to collection
    C.add_experiment(E2, condition = 'random_set_2')
    # Choose two features
    feature_0 = 'CD38'
    feature_1 = 'CD95'
    # Remove nan from all features
    C.clean_samples()
    # Compensate data
    C.compensate()
    # Normalize data
    #C.normalize(features = [feature_0, feature_1], mode = 'arcsinh')
    C.normalize(features = [feature_0, feature_1], mode = 'logicle')
    # Show experiment E1
    E1.show(feature_0, feature_1, pdf = pdf)
    # Show all experiments in the collection
    C.show(feature_0, feature_1, pdf = pdf)
    # Show scattered data, colored according to experiment index
    C.show_scatter([feature_0, feature_1], stride = 0, mode = 'experiments', pdf = pdf)
    # Show scattered data, colored according to condition index
    C.show_scatter([feature_0, feature_1], stride = 0, mode = 'conditions', pdf = pdf)
    # Show scattered data, colored according to labels
    C.show_scatter([feature_0, feature_1], stride = 0, mode = 'labels', pdf = pdf)
    # Show scattered data, colored according to density
    C.show_scatter([feature_0, feature_1], stride = 0, mode = 'density', pdf = pdf)
    # Show cluster distribution
    #C.show_distributions([feature_0, feature_1], pdf = pdf)

    pdf.close()
