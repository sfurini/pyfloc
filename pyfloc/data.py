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

import graphics
import settings

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
        print('Experiment {0:d} goes from sample {1:d} to sample {2:d}'.format(i_experiment, ind_start, ind_end))
        inds_experiment = []
        for ind in inds:
            if (ind >= ind_start) and (ind < ind_end):
                inds_experiment.append(ind-ind_start)
        return inds_experiment
    def compensate(self):
        for i_experiment, experiment in enumerate(self.experiments):
            print('Running compensation for experiment {0:d}'.format(i_experiment))
            experiment.compensate()
    def normalize(self, features, **kwargs):
        """
        Parameters
        ----------
        features list of str
            Name of the features to normalize
        """
        for feature in features:
            min_feature = self.get_min_feature(feature)
            max_feature = self.get_max_feature(feature)
            mean_feature = self.get_mean_feature(feature)
            std_feature = self.get_std_feature(feature)
            self.normalize_parameters[feature] = [min_feature, max_feature, mean_feature, std_feature]
            for i_experiment, experiment in enumerate(self.experiments):
                print('Running normalization for feature {0:s} in experiment {1:d} with mode {2:s}'.format(feature, i_experiment,kwargs.get('normalize_mode','min')))
                experiment.normalize(feature, min_feature, max_feature, mean_feature, std_feature, **kwargs)
    def convert_to_original_data(self, feature, data):
        """
        Parameters
        ----------
        feature: str
            Name of the feature
        data: np.ndarray
            One dimensional array with data along that feature
        """
        if feature in self.normalize_parameters.keys():
            return  self.normalize_parameters[feature][0] + data * (self.normalize_parameters[feature][1] - self.normalize_parameters[feature][0])
        else:
            return data
    def delete_samples(self, inds):
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
        print('Removing {0:d} samples'.format(len(inds)))
        inds_experiments = []
        for i_experiment, experiment in enumerate(self.experiments):
            inds_experiments.append(self.get_indexes_experiment(inds, i_experiment))
        # Two separate steps are necessary because indexes changes when samples are deleted
        for i_experiment, inds_experiment in enumerate(inds_experiments):
            print('Removing {0:d} samples from experiment {1:d}'.format(len(inds_experiment), i_experiment))
            self.experiments[i_experiment].delete_samples(inds_experiment)
        # Remove labels of deleted samples
        if len(inds) > 0:
            self.labels[inds] = -123456789 # just a random number to mark them for delete
            self.labels = self.labels[self.labels != -123456789]
    def clean_samples(self, feature, value):
        """
        Remove samples where feature is equal to value

        Parameters
        ----------
        feature: str
            The feature to check
        value:
            The value to remove
        """
        data = self.get_data_features([feature])
        if value == 'nan':
            inds_delete = np.where(np.isnan(data))[0]
        elif value == 'inf':
            inds_delete = np.where(np.isinf(data))[0]
        print('Removing {0:d} samples with feature {1:s} equal to {2:s}'.format(len(inds_delete),feature, str(value)))
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
    def remove_outliers(self, features, max_n_std = 5.0):
        print('Number of samples before outliers removal {0:d}'.format(self.get_n_samples()))
        for feature in features:
            data = self.get_data_features([feature])
            avr = np.mean(data)
            std = np.std(data)
            dist_avr = np.abs(data - avr)
            inds_delete = np.where(dist_avr > max_n_std*std)[0]
            print('Removing {0:d} outliers for feature {1:s}'.format(len(inds_delete),feature))
            self.delete_samples(inds_delete)
        print('Number of samples after outliers removal {0:d}'.format(self.get_n_samples()))
    def show_scatter(self, features, features_synonym = {}, stride = 0, contour = None, pdf = None):
        """
        Scatter plots of samples
        It works only for 2 features

        Parameters
        ----------
        features: list of str
            The names of the features
        features_synonym:   dict
            key: features
            values: alternative names
        stride: int
            Plotting period
        """
        if len(features) != 2:
            print('WARNING: scatter plot is possible only with two features')
            return
        features = deepcopy(features)
        data = self.get_data_features([features[0], features[1]])
        for i_feature, feature in enumerate(features):
            features[i_feature] = features_synonym.get(feature, feature)
        self.single_scatter_plot(data, self.labels, features, stride, contour, 'All Experiments', pdf)
        if (pdf is not None) and (len(self.get_conditions()) > 1): # if more than one conditions exist, make also separate plots (but only if we're plotting on PDF)
            for conditions in self.get_conditions():
                inds_conditions = self.get_indexes_conditions(conditions)
                self.single_scatter_plot(data[inds_conditions,:], self.labels[inds_conditions], features, stride, contour, conditions, pdf)
    def single_scatter_plot(self, data, data_colors, features, stride, contour, title, pdf):
        """
        It's the method actually making the scatter plot
        Use Collection.show_scatter as interface
        """
        if stride == 0:
            stride = max(int(data.shape[0] / 10000),1)
        data = data[::stride,:]
        data_colors = data_colors[::stride]
        f = plt.figure()
        ax = f.add_subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.plot(data[:,0], data[:,1],',k',markersize = 1.0, label = 'all samples')
        for data_color in set(data_colors):
            inds_data_color = (data_colors == data_color)
            ax.plot(data[inds_data_color,0], data[inds_data_color,1], '.', markersize = 2.0, color = settings.colors[data_color%len(settings.colors)], label = str(data_color))
        if contour is not None:
            ax.plot(contour[0], contour[1], 'o--k')
        #if density:
        #    density_peaks = bamboo.cluster.DensityPeaks(n_clusters = len(set(labels)),trajs = [data])
        #    dummy, rho = density_peaks.fit()
        #    rho = -np.log10(rho)
        #    rho[rho > np.min(rho)+2] = np.nan
        #    ax1.scatter(data[:,0], data[:,1], marker = ',', s = 1.0, c = rho, cmap = 'viridis') #, vmin = np.min(rho), vmax = np.min(rho)+4)
        plt.title(title)
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            self.fis = graphics.AxesScaleInteractor(f)

        from sklearn import mixture
        for data_color in set(data_colors):
            inds_data_color = (data_colors == data_color)
            if len(data[inds_data_color,0]) > 100: 
                H, xe, ye = np.histogram2d(np.log10(data[inds_data_color,0]), np.log10(data[inds_data_color,1]), bins = 200, range = [[1,4.5], [1,4]])
                xb = 0.5*(xe[:-1]+xe[1:])
                yb = 0.5*(ye[:-1]+ye[1:])
                X, Y = np.meshgrid(xb,yb)
                X = np.transpose(X)
                Y = np.transpose(Y)
                clf = mixture.GaussianMixture(n_components = 50, covariance_type = 'full')
                data_log10 = np.log10(data[inds_data_color,:])
                inds_del = np.where(np.sum(np.logical_not(np.isfinite(data_log10)), axis = 1))[0]
                data_log10 = np.delete(data_log10, inds_del, axis = 0)
                clf.fit(data_log10)
                XY = np.array([X.ravel(), Y.ravel()]).T
                Z = clf.score_samples(XY)
                Z = Z.reshape(X.shape)
                Z -= np.min(Z) 
                Z /= np.sum(Z)
                Zlog = np.log10(Z)
                Zlog[np.logical_not(np.isfinite(Zlog))] = np.nan

                f = plt.figure()
                ax = f.add_subplot(111)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.plot(data[:,0], data[:,1],',k',markersize = 1.0, label = 'all samples')
                ax.plot(data[inds_data_color,0], data[inds_data_color,1], '.', markersize = 2.0, color = settings.colors[data_color%len(settings.colors)], label = str(data_color))
                plt.title(str(data_color))
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
                pdf.savefig()
                plt.xscale('log')
                plt.yscale('log')
                pdf.savefig()
                plt.close()
                f = plt.figure()
                ax = f.add_subplot(111)
                ax.pcolormesh(X, Y, np.log10(H), cmap = plt.get_cmap('hot'))
                ax.contour(X, Y, Zlog, levels = np.linspace(np.nanmin(Zlog), np.nanmax(Zlog), 100), cmap = plt.get_cmap('winter'))
                pdf.savefig()
                plt.close()

    def show_distributions(self, features, features_synonym = {}, clusters_order = None, pdf = None):
        """
        """
        if len(set(self.labels)) == 1:
            print('WARNING: no distribution to plot if labels are not defined')
            return
        data = self.get_data_features(features)
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
        cax = ax.matshow(np.log10(average_features[clusters_order,:]), cmap = 'RdYlBu_r')
        cbar = f.colorbar(cax)
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
        Variables used to keep the interactive figures
    """
    def __init__(self, file_name = None, id_sample = 0, mode = 'all'):
        """
        Parameters
        ----------
        file_name: str
            Name of FCS file
        id_sample: int
        mode: str
            How to read the data:
                all = read everything
                random_INT = choose INT elements at random, this is usefull for quick tests of the code
        """
        self.sample = FCMeasurement(ID = id_sample, datafile = file_name)
        if mode == 'all':
            self.data = np.copy(self.sample.data.values[:,:])
        elif mode[0:7] == 'random_':
            n_samples_to_keep = int(mode[7:])
            inds_samples = np.random.choice(range(self.sample.data.values.shape[0]), size = n_samples_to_keep, replace = False)
            self.data = np.copy(self.sample.data.values[inds_samples,:])
        else:
            raise ValueError('ERROR: wrong reading mode {0:s}'.format(mode))
        self.data_norm = np.nan*np.ones(np.shape(self.data))
        self.fis = []
        print('Read {0:d} samples from {1:s}'.format(self.get_n_samples(), file_name))
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
            ind_features.append(self.get_features().index(feature))
        return ind_features
    def get_features(self):
        """
        Return
        ------
        list
            Name of the features
        """
        return list(self.sample.data.columns)
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
    def compensate(self):
        ind_features, compensate_matrix = self.get_compensation()
        self.data[:,ind_features] = np.dot(self.data[:,ind_features],np.linalg.inv(compensate_matrix))
    def normalize(self, feature, min_feature, max_feature, mean_feature, std_feature, **kwargs):
        """
        Parameters
        ---------
        feature:    str
        min_feature:    float
        max_feature:    float
        mean_feature:    float
        std_feature:    float

        normalize_mode: str
            See code for definition of the implemented normalization mode
        normalize_bias: float
        normalize_factor: float
            These are parameters that can be used to tune the normalization functions
        """
        ind_features = self.get_index_features([feature])
        if 'normalize_mode' not in kwargs:
            kwargs['normalize_mode'] = 'min'
        if kwargs['normalize_mode'] == 'min':
            self.data_norm[:,ind_features] = (self.data[:,ind_features] - min_feature)
        elif kwargs['normalize_mode'] == 'min_max':
            self.data_norm[:,ind_features] = (self.data[:,ind_features] - min_feature) / (max_feature - min_feature)
        elif kwargs['normalize_mode'] == 'mean_std':
            self.data_norm[:,ind_features] = (self.data[:,ind_features] - mean_feature) / std_feature
        elif kwargs['normalize_mode'] == 'arcsin':
            self.data_norm[:,ind_features] = np.arcsinh((self.data[:,ind_features] - kwargs.get('normalize_bias',0.0)*mean_feature) / kwargs.get('normalize_factor',5.0))
        elif kwargs['normalize_mode'] == 'arcsin_abs':
            if kwargs.get('normalize_bias',0.0) == 'min':
                dummy = (self.data[:,ind_features] - min_feature)
            else:
                dummy = (self.data[:,ind_features] - kwargs.get('normalize_bias',0.0))
            dummy[dummy < 0] = 0.0
            self.data_norm[:,ind_features] = np.arcsinh( dummy / kwargs.get('normalize_factor',5.0))
        else:
            raise NotImplementedError('ERROR normalization mode {0:s} not implemented'.format(kwargs['normalize_mode']))
    def delete_samples(self, inds):
        self.data = np.delete(self.data, inds, axis = 0)
        self.data_norm = np.delete(self.data_norm, inds, axis = 0)
    def show(self, feature_0, feature_1,  stride = 0, condition = None, pdf = None):
        """
        """
        if stride == 0:
            stride = max(int(self.get_n_samples() / 10000),1)
        data = self.get_data_features([feature_0, feature_1])
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.plot(data[::stride,0], data[::stride,1],'.',markersize = 1.0)
        plt.xlabel(feature_0)
        plt.ylabel(feature_1)
        if condition is not None:
            plt.title(condition)
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            self.fis.append(graphics.AxesScaleInteractor(f))
    def __str__(self):
        output  = 'Number of samples: {0:d}\n'.format(self.get_n_samples())
        output += 'Features: {0:s}\n'.format(','.join(self.get_features()))
        return output[:-1]

if __name__ == '__main__':
    print('---------------')
    print('Testing data.py')
    print('---------------')

    pdf = PdfPages('./test.pdf')

    E = Experiment('../examples/data/flowc/levine_13dim.fcs')
    C = Collection()
    C.add_experiment(E, condition = 'test1', labels = E.get_data_features('label'))
    C.add_experiment(E, condition = 'test2')
    #C.show_scatter(['CD4','CD8']) #, pdf = pdf)
    C.show_distributions(['CD4','CD8'], pdf = pdf)
    print(C)

    pdf.close()
