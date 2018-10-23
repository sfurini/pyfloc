#!/usr/bin/env python

import sys
import pickle
import functools
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import linear_sum_assignment
import graphics
import settings

plt.switch_backend('agg')

np.set_printoptions(linewidth = np.inf)
print = functools.partial(print, flush=True)

class Cluster(object):
    """
    Class for discretizing trajectories

    Classes inheriting Cluster should define:
        fit : method that initialize the clustering algorithm
            This method should define the attribute: cluster_analogic
        predict: method that take as input an analogic trajectory and return its discretized version

    Attributes
    ----------
    name: str
        Name of the clustering method
    trajs: list
        List of trajectories
        Each trajectory is a np.ndarray with shape: <number of samples> x <number of features>
    dtrajs: list 
        Discretized trajectories
        Each discretized trajectory is a np.ndarray with
            shape: <number of samples>
            values: index of the cluster for that sample
    clusters_analogic: np.ndarray with shape: <number of bins> x <number of features>
        Values used as representative of the clusters when back-converting to analogic values
    labels: list
        Reference labels
        Each element of the list is a np.ndarray with
            shape: <number of samples>
            values: label of the sample
    trajs_merged : np.ndarray
        Shape: <number of samples> x <number of features>
        All the trajectories combined into a single array
    dtrajs_merged: np.ndarray
        Shape: <number of samples>
        All the discretized trajectories combined into a single array
    labels_merged: np.ndarray
        Shape: <number of samples>
        All the labels combined into a single array
    """
    def __init__(self, trajs, labels = [], verbose = 0):
        """
        Parameters
        ----------
        trajs: list
            List of trajectories
            Each trajectory is a np.ndarray with shape: <number of samples> x <number of features>
        labels: list
            Reference labels
            Each element of the list is a np.ndarray with
                shape : <number of samples>
                values : label of the sample
        verbose: int
            How much output is produced
        """
        self.verbose = verbose
        self.trajs = trajs # analogical trajectories
        self.dtrajs = [] # discrete trajectories
        self.clusters_analogic = np.empty(0) # samples used as representative of clusters for analogic conversions
        self.name = 'Undefined' # name of the clustering algorithm
        self.fit_done = False
        #--- This is done to be sure that reference labels start from zero and are consecutives
        set_labels = set()
        for labels_traj in labels:
            for label in labels_traj:
                set_labels.add(label)
        list_labels = list(set_labels)
        list_labels.sort()
        dict_labels = {}
        for i, label in enumerate(list_labels):
            dict_labels[label] = i
        self.labels = []
        for labels_traj in labels:
            label_traj_new = []
            for label in labels_traj:
                label_traj_new.append(dict_labels[label])
            self.labels.append(np.array(label_traj_new))
        #--- Create merged trajectories / labels
        self.trajs_merged = np.vstack([traj for traj in self.trajs]).astype('float64')
        self.dtrajs_merged = -1*np.ones(self.n_frames()).astype(int)
        if len(labels) > 0:
            self.labels_merged = np.hstack([label for label in self.labels]).astype('int')
        else:
            self.labels_merged = None
    def n_trajs_analogic(self):
        """Return the number of analogical trajectories"""
        return len(self.trajs)
    def n_trajs(self):
        """Return the number of discretized trajectories"""
        return len(self.dtrajs)
    def n_dims(self):
        """Return the number of dimensions"""
        if not self.trajs:
            return np.nan
        list_n_dims = [self.n_dims_traj(traj) for traj in self.trajs]
        if all(list_n_dims[0] == other for other in list_n_dims):
            return list_n_dims[0]
        raise ValueError('ERROR: inconsistent dimensions')
    def n_dims_traj(self, traj):
        """Return the number of dimensions of the trajectory"""
        if len(np.shape(traj)) == 1:
            return 1
        else:
            return np.shape(traj)[1]
    def n_frames(self):
        """Return the total number of frames"""
        n = 0
        for traj in self.trajs:
            n += self.n_frames_traj(traj)
        return n
    def n_frames_traj(self, traj):
        """Return the number of frames of the trajectory"""
        return np.shape(traj)[0]
    def n_clusters(self):
        """Return the number of clusters"""
        if self.clusters_analogic.size > 0:
            return np.shape(self.clusters_analogic)[0]
        else:
            clusters = set()
            for dtraj in self.dtrajs:
                for cluster in dtraj:
                    clusters.add(cluster)
            return len(clusters)
    def set_labels(self):
        """Return the set of reference labels"""
        labels = set()
        for labels_traj in self.labels:
            for label in labels_traj:
                labels.add(label)
        return labels
    def n_labels(self):
        """Return the number of reference labels"""
        return len(self.set_labels())
    def get_index_merged(self, i_traj):
        """Return start and end index for trajetory i_traj in merged trajectory"""
        ind_start = 0
        ind_end = self.n_frames_traj(self.trajs[0])
        for i_traj in range(1, i_traj):
            ind_start += self.n_frames_traj(self.trajs[i_traj])
            ind_end += self.n_frames_traj(self.trajs[i_traj])
        return ind_start, ind_end
    def reset_trajectories(self, trajs = []):
        """Change the list of trajectories with trajs and reset dtrajs"""
        self.trajs = trajs
        self.dtrajs = []
    def transform(self):
        """Perform the discretization of all the trajectories trajectories"""
        self.dtrajs = []
        for i_traj, traj in enumerate(self.trajs):
            dtraj = self.predict(traj)
            self.dtrajs.append(dtraj)
            i_start, i_end = self.get_index_merged(i_traj)
            self.dtrajs_merged[i_start:i_end] =  dtraj
    def fit_predict(self, *args, **kwargs):
        if not self.fit_done:
            self.fit(*args, **kwargs)
        self.transform()
        if self.clusters_analogic.size == 0: # the clustering algorithm did not define clusters_analogic, so it's done here using the centroids
            self.clusters_analogic =self.centroids()
    def analogic(self, dtraj):
        """Return the discretized trajectory in analogic form"""
        if not isinstance(dtraj,np.ma.MaskedArray):
            dtraj = np.ma.masked_invalid(dtraj)
            dtraj = np.ma.masked_less(dtraj, 0)
        traj = np.nan*np.ones((len(dtraj),self.n_dims()))
        traj[np.logical_not(dtraj.mask),:] = self.clusters_analogic[dtraj.compressed().astype(int),:]
        return traj
    def antitransform(self):
        """Use the discretized trajectories to calculate the analogical trajectories"""
        trajs = []
        for dtraj in self.dtrajs:
            trajs.append(self.analogic(dtraj))
        self.trajs = trajs
    def get_mean_label(self, label):
        """Return mean for sample with this label"""
        return np.mean(self.trajs_merged[self.labels_merged == label,:], axis = 0)
    def get_std_label(self, label):
        """Return standard deviation for sample with this label"""
        return np.std(self.trajs_merged[self.labels_merged == label,:], axis = 0)
    def get_mean_cluster(self, cluster):
        """Return mean for sample of this cluster"""
        return np.mean(self.trajs_merged[self.dtrajs_merged == cluster,:], axis = 0)
    def get_std_cluster(self, cluster):
        """Return standard deviation for sample of this cluster"""
        return np.std(self.trajs_merged[self.dtrajs_merged == cluster,:], axis = 0)
    def n_samples(self, norm = False, list_clusters = None):
        """
        Calculate the number of samples for each clusters

        Parameters
        ----------
        norm: bool
            If True normalize the output to 1.0, otherwise return the total number of samples
        list_clusters: list
            If different from None calculate n_samples only for these clusters
        """
        if list_clusters is None:
            list_clusters = range(self.n_clusters())
        samples = np.zeros(len(list_clusters))
        for i, i_cluster in enumerate(list_clusters):
            for i_traj, dtraj in enumerate(self.dtrajs):
                n = np.sum(dtraj == i_cluster)
                if n:
                    samples[i] += int(n)
        if norm:
            samples /= np.sum(samples)
        return samples
    def n_samples_labels(self, list_labels = None, norm = False):
        """
        Calculate the number of samples for each label

        Parameters
        ----------
        list_clusters: list
            If different from None calculate n_samples only for these labels
        norm: bool
            If True normalize the output to 1.0, otherwise return the total number of samples
        """
        if list_labels is None:
            list_labels = list(self.set_labels())
        samples = np.zeros(len(list_labels))
        for i, label in enumerate(list_labels):
            for i_traj, label_traj in enumerate(self.labels):
                n = np.sum(label_traj == label)
                if n:
                    samples[i] += int(n)
        if norm:
            samples /= np.sum(samples)
        return samples
    def most_populated_cluster(self):
        """Return the index of the most populated cluster"""
        samples = self.n_samples()
        return np.argmax(samples)
    def dtrajs_index_sync(self, index_cluster = None):
        """
        Return a list with length equal to the number of dtrajs
        The elements of the list are integers equal to:
            - the first occurance of the cluster index_cluster in the corresponding discretized trajectory
            - -1 if the cluster doesn't occur
        If index_cluster is None the most populated one is used
        """
        if index_cluster is None:
            index_cluster = self.most_populated_cluster()
        indexes_sync = []
        for dtraj in self.dtrajs:
            indexes = np.where(dtraj == index_cluster)[0]
            if len(indexes) > 0:
                indexes_sync.append(indexes[0])
            else:
                indexes_sync.append(-1)
        return indexes_sync
    def remove_cluster(self, cluster):
        """Remove cluster from the discretized trajectories: samples after the first occurence of cluster are removed"""
        print('Number of states before removal {0:d}'.format(self.n_clusters()))
        inds_new_cluster = -1*np.ones(self.n_clusters()).astype(int)
        ind_new_cluster = 0
        clusters_analogic = []
        for ind_cluster, n_sample in enumerate(self.n_samples()):
            if ind_cluster != cluster:
                clusters_analogic.append(list(self.clusters_analogic[ind_cluster,:]))
                inds_new_cluster[ind_cluster] = ind_new_cluster
                ind_new_cluster += 1
        self.clusters_analogic = np.array(clusters_analogic)
        for i_traj, dtraj in enumerate(self.dtrajs):
            print('Occurances of cluster {0:d} in trajectory {1:d} = '.format(cluster, i_traj), np.where(dtraj == cluster)[0])
            while  len(np.where(dtraj == cluster)[0]):
                ind_first = np.where(dtraj == cluster)[0][0]
                print('Number of samples in dtraj {0:d} before removing cluster {1:d} = {2:d}'.format(i_traj, cluster, len(dtraj)))
                print('Dealing with occurance of cluster {0:d} at timestep {1:d} over {2:d}'.format(cluster, ind_first, len(dtraj)))
                if ind_first < 0.5*len(dtraj):
                    dtraj = dtraj[ind_first+1:]
                else:
                    dtraj = dtraj[:ind_first]
                print('Number of samples in dtraj {0:d} after removing cluster {1:d} = {2:d}'.format(i_traj, cluster, len(dtraj)))
            self.dtrajs[i_traj] = inds_new_cluster[dtraj]
        print('Number of states after removal {0:d}'.format(self.n_clusters()))
        self.remove_empty_states()
    def remove_empty_states(self):
        """Redefine the discretized trajectories if there are clusters without samples"""
        print('Number of states before removal {0:d}'.format(self.n_clusters()))
        inds_new_cluster = -1*np.ones(self.n_clusters()).astype(int)
        ind_new_cluster = 0
        clusters_analogic = []
        for ind_cluster, n_sample in enumerate(self.n_samples()):
            if n_sample > 0:
                clusters_analogic.append(list(self.clusters_analogic[ind_cluster,:]))
                inds_new_cluster[ind_cluster] = ind_new_cluster
                ind_new_cluster += 1
        self.clusters_analogic = np.array(clusters_analogic)
        for i_traj, dtraj in enumerate(self.dtrajs):
            self.dtrajs[i_traj] = inds_new_cluster[dtraj]
        print('Number of states after removal {0:d}'.format(self.n_clusters()))
    def get_outliers(self):
        dist_max = -np.inf
        for i_traj in range(self.n_trajs()):
            for  i_cluster in range(self.n_clusters()):
                frames = (self.dtrajs[i_traj] == i_cluster)
                dist = np.linalg.norm(self.trajs[i_traj][frames,:] - self.clusters_analogic[i_cluster,:], axis = 1)
                if np.size(dist):
                    ind_max = np.argmax(dist)
                    if dist[ind_max] > dist_max:
                        dist_max = dist[ind_max]
                        ind_max = np.where(self.dtrajs[i_traj] == i_cluster)[0][ind_max]
                        state_max = self.trajs[i_traj][ind_max]
        return state_max
    def distance_matrix(self, mode = 'euclidean'): 
        from sklearn.metrics import pairwise_distances
        if self.verbose > 0:
            print('Calculating distance matrix with metric {0:s}'.format(mode))
        if (mode == 'manhattan') or (mode == 'chebyshev'):
            dist_dims = np.zeros((self.n_dims(),self.n_frames(),self.n_frames()))
            for i_dim in range(self.n_dims()):
                dist_i = pairwise_distances(self.trajs_merged[:,i_dim].reshape(self.n_frames(),-1))
                dist_dims[i_dim,:,:] = dist_i
            if mode == 'manhattan':
                dist = np.sum(dist_dims, axis = 0)
            elif mode == 'chebyshev':
                dist = np.min(dist_dims, axis = 0)
        elif mode == 'euclidean':
            dist = pairwise_distances(self.trajs_merged)
        elif mode == 'angular':
            scalar = np.dot(self.trajs_merged, self.trajs_merged.transpose())
            teta = (scalar / np.sqrt(scalar.diagonal().reshape(scalar.shape[0],1)) / np.sqrt(scalar.diagonal().reshape(1,scalar.shape[0])))
            teta[teta > 1.0] = 1.0
            dist = np.arccos(teta)
        else:
            raise NotImplementedError('Method {0:s} does not exist'.format(mode))
        return dist
    def score(self):
        """
        Calculate the score of the clustering algorithm by comparing the clustering results with reference labels
        """
        n_clusters = self.n_clusters()
        n_labels = self.n_labels()
        precision_matrix = np.zeros((n_clusters, n_labels))
        count_matrix = np.zeros((n_clusters, n_labels))
        recall_matrix = np.zeros((n_clusters, n_labels))
        for i_cluster in range(n_clusters):
            for i_label in range(n_labels):
                detected = 0
                true = 0
                true_positive = 0
                for i_traj, dtraj in enumerate(self.dtrajs):
                    labels = self.labels[i_traj]
                    samples_i_cluster = (dtraj == i_cluster)
                    samples_i_label = (labels == i_label)
                    detected += np.sum(samples_i_cluster)
                    true += np.sum(samples_i_label)
                    true_positive += np.sum(samples_i_label*samples_i_cluster)
                if detected:
                    precision_matrix[i_cluster, i_label] = 1.0*true_positive/detected
                    count_matrix[i_cluster, i_label] = 1.0*true_positive
                if true:
                    recall_matrix[i_cluster, i_label] = 1.0*true_positive/true
        f_matrix = 2.0 * precision_matrix * recall_matrix / (precision_matrix + recall_matrix)
        f_matrix[np.isnan(f_matrix)] = 0.0
        if f_matrix.shape[0] > f_matrix.shape[1]:
            #inv_f = -1.0*np.transpose(count_matrix)
            inv_f = -1.0*np.transpose(f_matrix)
            pair_index_labels, pair_index_clusters = linear_sum_assignment(inv_f)
        else:
            #inv_f = -1.0*count_matrix
            inv_f = -1.0*f_matrix
            pair_index_clusters, pair_index_labels = linear_sum_assignment(inv_f)
        precision = np.zeros(n_labels)
        recall = np.zeros(n_labels)
        f = np.zeros(n_labels)
        for i_cluster, i_label in zip(pair_index_clusters, pair_index_labels):
            precision[i_label] = precision_matrix[i_cluster, i_label]
            recall[i_label] = recall_matrix[i_cluster, i_label]
            f[i_label] = f_matrix[i_cluster, i_label]
        #print('Count matrix: ',count_matrix)
        #print('F-matrix: ',f_matrix)
        #print('Pair index labels: ',pair_index_labels)
        #print('Pair index clusters: ',pair_index_clusters)
        #print('Precision: ',precision,', average = ',np.mean(precision[pair_index_labels]))
        #print('Recall: ',recall,', average = ',np.mean(recall[pair_index_labels]))
        #print('f: ',f,', average = ',np.mean(f[pair_index_labels]))
        #print('Score clustering algorithm: {0:f} +/- {1:f}'.format(np.mean(f[pair_index_labels]),np.std(f[pair_index_labels])))
        return pair_index_labels, pair_index_clusters, precision[pair_index_labels], recall[pair_index_labels], f[pair_index_labels]
    def centroids(self):
        n_clusters = self.n_clusters()
        clusters_analogic = np.empty((n_clusters, self.n_dims()))
        for i_cluster in range(n_clusters):
            if self.verbose > 0:
                print('Computing centroid for cluster {0:d} over {1:d}'.format(i_cluster,n_clusters))
            cluster_data = np.empty((0,self.n_dims()))
            for i_traj, dtraj in enumerate(self.dtrajs):
                inds = dtraj == i_cluster
                if np.sum(inds) > 0:
                    cluster_data = np.vstack((cluster_data, self.trajs[i_traj][inds,:]))
            clusters_analogic[i_cluster,:] = np.mean(cluster_data, axis = 0)
        return clusters_analogic
    def sum_values(self, samples, values):
        """
        Sums the values <values> of samples <samples> according to the cluster they belong to

        Returns
        -------
        np.array : <number of clusters>
            values : The sum of <values> for <samples> beloning to each cluster
        """
        dsamples = self.discretize(samples)
        values = np.ma.masked_invalid(values)
        sums = np.zeros(self.grid.n_clusters())
        for i_cluster in range(self.grid.n_clusters()):
            sums[i_cluster] = np.sum(values[dsamples == i_cluster])
        return sums
    def show_n_samples(self, pdf = None):
        """
        Plot the number of samples
        """
        if self.n_dims() == 1:
            f = plt.figure()
            ax = f.add_subplot(111)
            x = self.clusters_analogic.reshape(self.n_clusters())
            ind_sort = np.argsort(x)
            ax.plot(x[ind_sort], self.n_samples()[ind_sort],'o-')
            plt.xlabel('Reaction Coordinate 0')
            plt.ylabel('Number of samples')
            if pdf is not None:
                pdf.savefig()
                plt.close(f)
        elif self.n_dims() == 2:
            f = plt.figure()
            ax = f.add_subplot(111, projection = '3d')
            ax.scatter(self.clusters_analogic[:,0], self.clusters_analogic[:,1], self.n_samples())
            plt.xlabel('Reaction Coordinate 0')
            plt.ylabel('Reaction Coordiante 1')
            if pdf is not None:
                pdf.savefig()
                plt.close(f)
        else:
            raise ValueError('ERROR: not implemented for {0:d} dimensions'.format(self.n_dims()))
    def show(self, pdf = None, plot_trajs = False, plot_maps = True, plot_labels = False, stride = 0):
        """
        plot_trajs: bool
            Show plots of single trajectories in analogical and digital form
        plot_maps:  bool
            Show samples in 2D spaces 
        plot_labels:    bool
            Show distributions of parameters over labels
        """
        #plt.rcParams['image.cmap'] = 'Paired'
        if (not plot_trajs) and (self.n_dims() == 1):
            print('WARNING: nothing to do for 1D data if plot_traj is False')
            return
        if plot_labels and (self.labels is None):
            raise ValueError('ERROR: cannot plot labels if they are None')
        if stride == 0:
            stride = max(int(self.n_frames() / 10000),1)
        if plot_trajs:
            for i_traj, traj in enumerate(self.trajs):
                traj = self.trajs[i_traj]
                if len(self.dtrajs):
                    dtraj = self.dtrajs[i_traj]
                    dtraj_analogic = self.analogic(dtraj)
                else:
                    dtraj_analogic = None
                f = plt.figure()
                for i_dim in range(self.n_dims()):
                    ax = f.add_subplot(self.n_dims(),1,i_dim+1)
                    if i_dim == 0:
                        plt.title('Trajectory '+str(i_traj))
                    ax.plot(traj[:,i_dim],'-b', label = 'original trajectory')
                    if dtraj_analogic is not None:
                        ax.plot(dtraj_analogic[:,i_dim],'-r',label = 'analogical conversion')
                plt.legend()
                if pdf is not None:
                    pdf.savefig()
                    plt.close(f)
        if plot_maps:
            for i in range(self.n_dims()-1):
                for j in range(i+1,self.n_dims()):
                    f = plt.figure()
                    if self.labels is not None:
                        ax1 = f.add_subplot(2,1,1)
                        plt.title(self.name)
                        ax2 = f.add_subplot(2,1,2)
                    else:
                        ax1 = f.add_subplot(1,1,1)
                        plt.title(self.name)
                    for i_traj, traj in enumerate(self.trajs):
                        ax1.plot(traj[::stride,i],traj[::stride,j],'.k')
                        if len(self.dtrajs):
                            dtraj = self.dtrajs[i_traj]
                            dtraj_analogic = self.analogic(dtraj)
                            for i_cluster in range(self.n_clusters()):
                                if len(np.shape(dtraj)) == 1:
                                    #inds_cluster = (dtraj == i_cluster)
                                    #ax1.plot(traj[inds_cluster,i][::stride],traj[inds_cluster,j][::stride],'.',color = settings.colors[i_cluster%len(settings.colors)])
                                    inds_cluster = (dtraj[::stride] == i_cluster)
                                    ax1.plot(traj[::stride,:][inds_cluster,i],traj[::stride,:][inds_cluster,j],'.', markersize = 1.0, color = settings.colors[i_cluster%len(settings.colors)])
                                    ax1.plot(self.clusters_analogic[i_cluster,i],self.clusters_analogic[i_cluster,j],'o',color=settings.colors[i_cluster%len(settings.colors)], markeredgecolor = 'k', label = str(i_cluster))
                        if self.clusters_analogic.size > 0:
                            for i_cluster in range(self.n_clusters()):
                                ax1.plot(self.clusters_analogic[i_cluster,i],self.clusters_analogic[i_cluster,j],'o',color=settings.colors[i_cluster%len(settings.colors)], markeredgecolor = 'k')
                        if self.labels:
                            ax2.scatter(traj[::stride,i],traj[::stride,j],c=self.labels[i_traj][::stride].astype(int), vmin = 0, vmax = 10)
                        else:
                            ax2.plot(traj[::stride,i],traj[::stride,j],',k')
                    plt.sca(ax1)
                    plt.legend()
                    plt.ylabel('Feature {0:d}'.format(j))
                    plt.sca(ax2)
                    plt.xlabel('Feature {0:d}'.format(i))
                    plt.ylabel('Feature {0:d}'.format(j))
                    if pdf is not None:
                        pdf.savefig()
                        plt.close()
        if plot_labels:
            means_labels = np.empty((self.n_labels(),self.n_dims()))
            for label in range(self.n_labels()):
                f = plt.figure()
                ax1 = f.add_subplot(1,1,1)
                means_labels[label,:] =  self.get_mean_label(label)
                ax1.errorbar(range(self.n_dims()), means_labels[label,:], yerr = self.get_std_label(label), fmt = 'o', label = label)
                plt.xlabel('Feature')
                plt.ylabel('mean +/- std')
                plt.legend()
                if pdf is not None:
                    pdf.savefig()
                    plt.close()
            #plt.rcParams['image.cmap'] = 'bwr'
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            cax = ax.matshow(means_labels, cmap = 'bwr')
            f.colorbar(cax)
            plt.xlabel('Features')
            plt.ylabel('Labels')
            if pdf is not None:
                pdf.savefig()
                plt.close()
    def __str__(self):
        output = 'Results of clustering with method {0:s}\n'.format(self.name)
        n_samples_per_cluster = self.n_samples()
        labels_found = []
        if self.labels is not None:
            n_samples_per_label = self.n_samples_labels()
            for i, label in enumerate(np.argsort(n_samples_per_label)):
                output += '\tLabel {0:d} = {1:d}\n'.format(int(label), int(n_samples_per_label[label]))
                output += '\t\tmeans = ' + str(self.get_mean_label(label)) + '\n'
                output += '\t\tstds = ' + str(self.get_std_label(label)) + '\n'
            pairing_labels, pairing_clusters, precision, recall, f = self.score()
        for i_cluster in range(self.n_clusters()):
            output += '\tCluster {0:d}\n'.format(i_cluster)
            output += '\t\tnumber of samples = {0:d}\n'.format(int(n_samples_per_cluster[i_cluster]))
            output += '\t\tmeans = ' + str(self.get_mean_cluster(i_cluster)) + '\n'
            output += '\t\tstds = ' + str(self.get_std_cluster(i_cluster)) + '\n'
            if self.labels is not None:
                if self.n_labels() >= self.n_clusters():
                    output += '\t\tpairing label[{0:d}] = {1:d}\n'.format(pairing_labels[i_cluster], int(n_samples_per_label[pairing_labels[i_cluster]]))
                    output += '\t\tprecision = {0:f}\n'.format(precision[i_cluster])
                    output += '\t\trecall = {0:f}\n'.format(recall[i_cluster])
                    output += '\t\tf-score = {0:f}\n'.format(f[i_cluster])
                    if f[i_cluster] > 0.2:
                        labels_found.append(pairing_labels[i_cluster])
                else:
                    if i_cluster in pairing_clusters:
                        i_label = np.where(pairing_clusters == i_cluster)[0][0]
                        label = pairing_labels[i_label]
                        output += '\t\tpairing label[{0:d}] = {1:d}\n'.format(label, int(n_samples_per_label[label]))
                        output += '\t\tprecision = {0:f}\n'.format(precision[i_label])
                        output += '\t\trecall = {0:f}\n'.format(recall[i_label])
                        output += '\t\tf-score = {0:f}\n'.format(f[i_label])
                        if f[i_label] > 0.2:
                            labels_found.append(label)
        if self.labels:
            labels_found.sort()
            output += '\tLabels found: '+str(labels_found)+'\n'
            output += '\tAverage f-score = {0:f} +/- {1:f}\n'.format(np.mean(f), np.std(f))
        return output[:-1]
    def dump(self, file_name):
        with open(file_name,'wb') as fout:
            pickle.dump(self, fout)

class Unique(Cluster):
    """
    Class for clustering each unique point in its own cluster
    N.B. It works only when feature values are integer numbers

    Attributes
    ----------
    list_samples: list
        List of all the unique values
    """
    def __init__(self, trajs, labels = None):
        super(Unique,self).__init__(trajs, labels)
        self.list_samples = []
        self.name = 'Single Points'
    def fit(self):
        print('Searching for set of all possible unique samples')
        set_samples = set()
        for traj in self.trajs:
            for i_frame in range(self.n_frames_traj(traj)):
                sample = tuple(traj[i_frame,:].astype(int))
                set_samples.add(sample)
        self.list_samples = []
        for sample in set_samples:
            self.list_samples.append(list(sample))
        self.clusters_analogic = np.array(self.list_samples)
        print('Number of unique samples = ',len(set_samples))
    def predict(self, traj):
        dtraj = []
        for i_frame in range(self.n_frames_traj(traj)):
            dtraj.append(self.list_samples.index(list(traj[i_frame,:])))
        return np.array(dtraj)

class ClusterSKlearn(Cluster):
    """
    Attributes
    ----------
    """
    def __init__(self, trajs = None, labels = [], verbose = 0):
        super(ClusterSKlearn, self).__init__(trajs, labels, verbose)
    def predict(self, traj):
        return self.algorithm.predict(traj)

class Kmeans(ClusterSKlearn):
    def __init__(self, trajs = None, labels = [], verbose = 0):
        from sklearn.cluster import KMeans
        self.algorithm = KMeans
        super(Kmeans, self).__init__(trajs, labels, verbose)
        self.name = 'Kmeans'
    def fit(self, n_clusters, *args, **kwargs):
        print('Clustering data with Kmeans algorithm')
        self.algorithm = self.algorithm(n_clusters = n_clusters, n_init = 1000, init = 'k-means++', max_iter = 10000)
        self.algorithm.fit(self.trajs_merged)
        self.clusters_analogic = self.algorithm.cluster_centers_
        self.fit_done = True

class MiniBatchKmeans(ClusterSKlearn):
    def __init__(self, trajs = None, labels = None):
        from sklearn.cluster import MiniBatchKMeans
        self.algorithm = MiniBatchKMeans
        super(MiniBatchKmeans, self).__init__(trajs, labels)
        self.name = 'MiniBatchKmeans'
    def fit(self, n_clusters, *args, **kwargs):
        print('Clustering data with MiniBatchKmeans algorithm')
        self.algorithm = self.algorithm(n_clusters = n_clusters, n_init = 10, init = 'k-means++', max_iter = 300, batch_size = 100)
        self.algorithm.fit(self.trajs_merged)
        self.clusters_analogic = self.algorithm.cluster_centers_

class Birch(ClusterSKlearn):
    def __init__(self, trajs = None, labels = None):
        from sklearn.cluster import Birch as BIrch
        self.algorithm = BIrch
        super(Birch, self).__init__(trajs, labels)
        self.name = 'Birch'
    def fit(self, n_clusters, *args, **kwargs):
        print('Clustering data with Birch algorithm')
        self.algorithm = self.algorithm(n_clusters = n_clusters, threshold = 0.5, branching_factor = 50, compute_labels = True)
        self.algorithm.fit(self.trajs_merged)

class Agglomerative(ClusterSKlearn):
    def __init__(self, trajs = None, labels = None):
        from sklearn.cluster import AgglomerativeClustering
        self.algorithm = AgglomerativeClustering
        super(Agglomerative, self).__init__(trajs, labels)
        self.name = 'Agglomerative'
    def fit(self, n_clusters, *args, **kwargs):
        print('Clustering data with Agglomerative algorithm')
        self.algorithm = self.algorithm(n_clusters = n_clusters, linkage = 'average')
        self.algorithm.fit(self.trajs_merged)
    def predict(self, traj):
        for i_traj_internal, traj_internal in enumerate(self.trajs): # search for traj among self.trajs
            if self.n_frames_traj(traj_internal) == len(traj):
                if np.all(np.prod(traj_internal == traj, axis = 1).astype(bool)): # this is the internal trajectory
                    i_start, i_end = self.get_index_merged(i_traj_internal)
                    return np.array(self.algorithm.labels_[i_start:i_end])
        raise ValueError('ERROR: missing trajectory')

class AffinityPropagation(ClusterSKlearn):
    def __init__(self, trajs = None, labels = None):
        from sklearn.cluster import AffinityPropagation as AFP
        self.algorithm = AFP
        super(AffinityPropagation, self).__init__(trajs, labels)
        self.name = 'Affinity Propagation'
    def fit(self, *args, **kwargs):
        print('Clustering data with AffinityPropagation algorithm')
        self.algorithm = self.algorithm(damping = 0.5, max_iter = 1000)
        self.algorithm.fit(self.trajs_merged)

class MeanShift(ClusterSKlearn):
    def __init__(self, trajs = None, labels = None):
        from sklearn.cluster import estimate_bandwidth
        from sklearn.cluster import MeanShift as MSH
        super(MeanShift, self).__init__(trajs, labels)
        bandwidth = estimate_bandwidth(self.trajs_merged, quantile = 0.3, n_samples = 500)
        self.algorithm = MSH
        self.name = 'MeanShift'
    def fit(self, bandwidth, *args, **kwargs):
        print('Clustering data with MeanShift algorithm, bandwidth = ', bandwidth)
        self.algorithm = self.algorithm(bandwidth = bandwidth)
        self.algorithm.fit(self.trajs_merged)
        self.clusters_analogic = self.algorithm.cluster_centers_

class GaussianMixture(ClusterSKlearn):
    def __init__(self, trajs = None, labels = None):
        from sklearn.mixture import GaussianMixture as GM
        self.algorithm = GM
        super(GaussianMixture, self).__init__(trajs, labels)
        self.name='GaussianMixture'
    def fit(self, n_clusters, *args, **kwargs):
        print('Clustering data with Gaussian Mixture algorithm')
        self.algorithm = self.algorithm(n_components = n_clusters, covariance_type = 'full')
        self.algorithm.fit(self.trajs_merged)
        self.clusters_analogic = self.algorithm.means_

#class DBSCAN(ClusterSKlearn):
#	def fit(self):
#		self.predict_discrimination=1
#		from sklearn.cluster import DBSCAN
#		from sklearn.preprocessing import StandardScaler
#		self.data = StandardScaler().fit_transform(self.data)
#		print '\n\nClustering data with DBSCAN...'
#		self.algorithm = DBSCAN()
#		self.algorithm.fit(self.data)
#		self.name='DBSCAN'
 
#class HDBSCAN(ClusterSKlearn):
#    def fit(self, min_cluster_size = 100, min_samples = 10):
#        import hdbscan
#        print 'Clustering data with HDBSCAN algorithm...'
#        self.algorithm = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples = min_samples)
#        self.algorithm.fit(self.trajs_merged)
#	self.name='HDBSCAN'
#    def run(self):
#        dtraj = self.algorithm.labels_
#        self.bin_centers = np.empty((len(set(dtraj)),self.n_dims()))
#        for i_cluster in range(len(set(dtraj))):
#            self.bin_centers[i_cluster,:] = np.mean(self.trajs[0][dtraj == i_cluster,:],axis = 0)
#        self.dtrajs = [dtraj]
#
#
#class Ward(ClusterSKlearn):
## No centers e non ha predict
#    def fit(self, nclusters):
#	self.predict_discrimination=1
#        from sklearn.cluster import AgglomerativeClustering
#	from sklearn.neighbors import kneighbors_graph
#        print 'Clustering data with Ward algorithm...'
#        self.algorithm = AgglomerativeClustering(n_clusters=nclusters, linkage='ward')
#        self.algorithm.fit(self.data)
#	self.name='Ward'
#
#
#class SpectralClustering(ClusterSKlearn):
#	def fit(self, nclusters):
#		self.predict_discrimination = 1
#		from sklearn.cluster import SpectralClustering
#		print 'Clustering data with Spectral Clustering algorithm...'
#		self.algorithm = SpectralClustering(n_clusters=nclusters)
#		self.algorithm.fit(self.data)
#		self.name='SpectralClustering'
### MM inizio
class KDPKNN(Cluster):
    def __init__(self, n_clusters, trajs = None, ref_labels = None, percent = 1.0):
        if ref_labels is not None:
            ref_labels = [np.hstack([ref_label for ref_label in ref_labels]).astype('int')]
        super(KDPKNN,self).__init__(trajs, ref_labels)
        self.label = -1*np.ones(self.n_frames(), dtype = np.int)
        self.n_clusters = n_clusters
        self.percent = percent
	#self.min_samples_per_region = min_samples_per_region
    def run(self, pdf = None):
	#k = int(np.sqrt(len(self.trajs[0])))
        k = int(len(self.trajs[0])**(1.0/3.0))
        k = np.minimum(k,100)
        print ("k = ", k)
        min_samples_per_region = np.maximum(50 - k, 10) #piu' regioni ci sono, meno punti e' necessario prendere
        n_clusters_DP_per_region = 5
        C = run('Kmeans', trajs = self.trajs, n_clusters = k, pdf = pdf)
        #C = Kmeans(trajs = self.trajs, labels = self.labels)
        #C.fit(n_clusters = k)       
        u = np.unique(C.dtrajs[0])
        taken = []
        for i in range(len(u)):
            i_taken = np.where(C.dtrajs[0] == u[i])[0]
            data = [self.trajs[0][i_taken]] 
            D = run('DensityPeaks', trajs = data, n_clusters = n_clusters_DP_per_region, percents = 1.0, pdf = pdf)
            #dummy, dummy2, i_cluster_centers = D.fit(pdf) #oltre ai centri dei cluster fatti restituire altri 2 punti appartenenti a quel cluster
            i_cluster_centers = D.ind_cluster_centers
            labels = D.dtrajs[0]
            labels[i_cluster_centers] = -1
            points_per_label = []
            for j in range(len(i_cluster_centers)):
                i_labels = np.where(labels==j)[0]
                i_min = np.minimum(len(i_labels), min_samples_per_region)
                points_per_label.append(i_labels[0:i_min])
            points_list = [val for sublist in points_per_label for val in sublist]
            taken.append(i_taken[i_cluster_centers]) #indice dei centri dei cluster
            taken.append(i_taken[points_list])
        taken = [val for sublist in taken for val in sublist]	
        data_final_step = [self.trajs[0][taken]]
        training = run('DensityPeaks', trajs = data_final_step, n_clusters =self.n_clusters, pdf = pdf) 
        labels = training.dtrajs[0]
        from sklearn.neighbors import KNeighborsClassifier
        print ('Clustering data with KNN algorithm...')
        clustering = KNeighborsClassifier()
        clustering.fit(data_final_step[0], labels)
        dtraj = clustering.predict(self.trajs[0])
        self.bin_centers = np.empty((len(set(dtraj)),self.n_dims()))
        for i_cluster in range(len(set(dtraj))):
            self.bin_centers[i_cluster,:] = np.mean(self.trajs[0][dtraj == i_cluster,:],axis = 0)
        self.dtrajs = [dtraj]
### MM fine

class DensityPeaks(ClusterSKlearn):
    """
    Attributes
    ----------
    dist:   np.ndarray
        Shape: <number of samples> x <number of samples>
        Matrix of pairwise distances
    rho: np.ndarray
        Shape: <number of samples>
        Densities
    delta: np.ndarray
        Shape: <number of samples>
        Distances from higher density samples
    nneigh: np.ndarray
        Shape: <number of samples>
        Indexes of the closest point with higher density
    ind_cluster_centers:    list
        Indexes of the samples that are used as cluster centers
    manual_clusters_selector: None / graphics.PointInteractor
        This is different from None when clusters are selected manually
    kernel_radius:  float
        The radius of the kernel
    """
    def __init__(self, trajs = None, labels = [], verbose = 0):
        super(DensityPeaks, self).__init__(trajs, labels, verbose)
        self.dist = np.empty((self.n_frames(), self.n_frames()))
        self.rho = np.empty(self.n_frames())
        self.delta = np.empty(self.n_frames())
        self.nneigh = np.empty(self.n_frames()).astype(int)
        self.ind_cluster_centers = []
        self.manual_clusters_selector = None
        self.kernel_radius = -1.0
        self.name = 'Density Peaks'
    def get_energy(self, percent, n_stds_delta, ns_clusters, pdf, n_gaussians = 1):
        """
        It finds the best candidates as cluster centers and return the delta energy for this clustering scheme

        See DensityPeaks.fit for parameter definition

        n_gaussians int
            Number of gaussians used to fit delta

        Return
        ------
        float
            The energy
        """
        from hiprec_erf import hiprec_erf
        from sklearn.mixture import GaussianMixture as GM
        if isinstance(ns_clusters, float):
            ns_clusters = int(ns_clusters)
        mask = np.logical_not(np.eye(self.dist.shape[0]).astype('bool'))
        #self.kernel_radius = np.percentile(self.dist[mask], percent)
        self.kernel_radius = percent
        if self.kernel_radius <= 0.0:
            self.ind_cluster_centers = []
            return -np.inf
        #--- Computing density (rho)
        if self.verbose > 0:
            print('Computing densities using kernel with radius {0:12.6f}'.format(self.kernel_radius))
        if ns_clusters is None:
            if self.verbose > 0:
                print('Considering as cluster centers the samples above {0:f} stds along delta'.format(n_stds_delta))
        elif isinstance(ns_clusters,int):
            if self.verbose > 0:
                print('Selecting {0:d} clusters'.format(ns_clusters))
        elif isinstance(ns_clusters,list):
            ns_clusters = [int(n_clusters) for n_clusters in ns_clusters] # to be sure that they're all integers
        #--- Gaussian kernel
        kernel_matrix = np.exp(-np.power(self.dist/self.kernel_radius,2.0))
        kernel_matrix[np.eye(kernel_matrix.shape[0]).astype(bool)] = 0.0 # this is to skip the distance from itself
        #--- Square kernel
        #kernel_matrix = (self.dist < self.kernel_radius)
        #--- Sum for all samples
        self.rho = np.sum(kernel_matrix, axis = 1)
        #--- Computing distances from higher density samples (delta)
        if self.verbose > 0:
            print('Searching neighbours')
        ordrho = np.argsort(self.rho)[-1::-1]
        rho_sorted = self.rho[ordrho]
        self.nneigh[ordrho[0]] = ordrho[0] # arbitrary convention: for the point with highest density, the neighbour at with higher density is itself
        self.delta[ordrho[0]] = np.max(self.dist[ordrho[0],:]) # arbitrary convention: for the point with highest density set delta to the maximum distance of this point from other samples
        for i in range(1,self.n_frames()):
            # ordrho[i] is the index of the sample, in this way the samples are analyzed in order of decreasing density
            # ordrho[:i] are the indexes of all the samples with densities higher than sample ordrho[i]
            i_closest_sample_higher_density = ordrho[np.argmin(self.dist[ordrho[i],ordrho[:i]])] # index of the same at minimum distance from sample ordrho[i] among the ones with with higher densities
            self.delta[ordrho[i]] = self.dist[ordrho[i], i_closest_sample_higher_density]
            self.nneigh[ordrho[i]] = i_closest_sample_higher_density  # the neighbour is the closest point among the ones with higher density
        rho_norm = (self.rho - np.min(self.rho)) / (np.max(self.rho) - np.min(self.rho))
        delta_norm = (self.delta - np.min(self.delta)) / (np.max(self.delta) - np.min(self.delta))
        #--- Plotting rho-delta relation
        if pdf is not None:
            f = plt.figure()
            ax1 = f.add_subplot(2,1,1)
            plt.title('Percentile {0:f}'.format(percent))
            ax2 = f.add_subplot(2,1,2)
            ax1.plot(self.rho,self.delta,'.k', markersize = 1.0)
            plt.ylabel('delta')
            ax2.plot(rho_norm,delta_norm,'.k', markersize = 1.0)
            plt.xlabel('rho_norm')
            plt.ylabel('delta_norm')
            plt.xscale('log')
            plt.yscale('log')
            pdf.savefig()
            plt.close()
        #--- Selecting the centers of the clusters
        rho_norm_log = np.log10(rho_norm)
        delta_norm_log = np.log10(delta_norm)
        inds_finite = np.isfinite(rho_norm_log*delta_norm_log)
        probs = np.inf*np.ones(self.n_frames())
        probs_gmm = np.inf*np.ones(self.n_frames(), dtype = 'float64')
        rho_edges = np.linspace(np.min(rho_norm_log[np.isfinite(rho_norm_log)]), np.max(rho_norm_log[np.isfinite(rho_norm_log)]), 20)
        ind_cluster_centers = [] # here, the indexes of the sample in low probability regions are stored
        #------ Calculate average values and standard deviations of deltas along rho
        means_deltas = []
        stds_deltas = []
        rho_bins = []
        gaussian_mixture_models = [] # vettore degli oggetti gaussian mixture
        for ir in range(len(rho_edges)-1):
            inds_sample_bin = (rho_norm_log > rho_edges[ir]) * (rho_norm_log <= rho_edges[ir+1]) * np.isfinite(rho_norm_log) * np.isfinite(delta_norm_log)
            if np.sum(inds_sample_bin) > 10: # check if there are enough samples in the bin
                rho_bins.append(0.5*(rho_edges[ir]+rho_edges[ir+1]))
                delta_in_bins = delta_norm_log[inds_sample_bin]
                means_deltas.append(np.mean(delta_in_bins))
                stds_deltas.append(np.std(delta_norm_log[inds_sample_bin]))
                #2) percentile above mean
                #deltas_above_mean = delta_in_bins[delta_in_bins > means_deltas[-1]] - means_deltas[-1]
                #std_deltas = np.percentile(deltas_above_mean, 68.27)
                #stds_deltas.append(std_deltas)
                #delta_in_bins = delta_in_bins[delta_in_bins > means_deltas[-1]]
                delta_in_bins = delta_in_bins[delta_in_bins < np.percentile(delta_in_bins, 90)]
                if len(delta_in_bins) < 100:
                    gaussian_mixture_models.append(GM(n_components = 1, covariance_type = 'full').fit(delta_in_bins.reshape(-1,1)))
                else:
                    gaussian_mixture_models.append(GM(n_components = n_gaussians, covariance_type = 'full').fit(delta_in_bins.reshape(-1,1)))
        if len(rho_bins) < 2:
            self.ind_cluster_centers = []
            return -np.inf
        rho_norm_log[np.logical_not(np.isfinite(rho_norm_log))] = np.nan
        ind_min = np.argsort(rho_norm_log)[0]
        f_means = sp.interpolate.interp1d([rho_norm_log[ind_min],]+rho_bins, [delta_norm_log[ind_min],]+means_deltas, bounds_error = False, fill_value = 'extrapolate')
        f_stds = sp.interpolate.interp1d(rho_bins, stds_deltas, bounds_error = False, fill_value = (stds_deltas[0], stds_deltas[-1]))
        rho_bins = np.array(rho_bins)
        rho_delta = np.min(rho_bins[1:] - rho_bins[:-1])
        if pdf is not None:
            f = plt.figure()
            ax = f.add_subplot(111)
            plt.title('Percentile {0:f}'.format(percent))
            ax.plot(rho_norm_log, delta_norm_log,'.k', markersize = 1.0)
            rho_th = np.linspace(np.min(rho_norm_log[np.isfinite(rho_norm_log)]),np.max(rho_norm_log[np.isfinite(rho_norm_log)]),100)
            ax.plot(rho_th, f_means(rho_th), '-r')
            if n_stds_delta is not None:
                ax.plot(rho_th, f_means(rho_th)+n_stds_delta*f_stds(rho_th), ':r')
            ax.errorbar(rho_bins, means_deltas, yerr = stds_deltas)
            for i_gmm, gmm in enumerate(gaussian_mixture_models):
                delta_th = np.linspace(np.min(delta_norm_log[np.isfinite(delta_norm_log)]), 0.0, 1000)
                #print('rho_delta = ',rho_delta)
                #print('delta_th = ',delta_th)
                gauss_th = gmm.score_samples(delta_th.reshape(-1,1))
                gauss_th = np.exp(gauss_th)
                #print('gauss_th = ',gauss_th)
                #print('cov = ',gmm.covariances_)
                gauss_th = rho_bins[i_gmm] + rho_delta * ( -0.5 +  (gauss_th - np.min(gauss_th)) / (np.max(gauss_th) - np.min(gauss_th)) )
                ax.plot(gauss_th, delta_th, '-g')
                for i_gauss in range(gmm.n_components):
                    ax.plot(rho_bins[i_gmm], gmm.means_[i_gauss], 'og')
            plt.xlabel('rho_norm_log')
            plt.ylabel('delta_norm_log')
            pdf.savefig()
            plt.close() 
        #------ Search for outliers
        for i_sample in range(self.n_frames()):
            if delta_norm_log[i_sample] > f_means(rho_norm_log[i_sample]):
                # 1) compute prob(delta > mu) assuming gaussian 
                mu = f_means(rho_norm_log[i_sample])
                sigma = f_stds(rho_norm_log[i_sample])
                x = delta_norm_log[i_sample]
                probs[i_sample] =  0.5*(1.0 - hiprec_erf( float((x - mu) / (np.sqrt(2)*sigma) ) ) )
                if n_stds_delta is not None:
                    if delta_norm_log[i_sample] > f_means(rho_norm_log[i_sample]) + n_stds_delta*f_stds(rho_norm_log[i_sample]):
                        ind_cluster_centers.append(i_sample)
                # 2) compute prob(delta > mu) using gaussianmixtures
                if np.min(np.abs(rho_norm_log[i_sample] - rho_bins)) < (rho_edges[1]-rho_edges[0]):
                    i_gmm = np.argmin(np.abs(rho_norm_log[i_sample] - rho_bins))
                    gmm = gaussian_mixture_models[i_gmm]
                    probs_gmm[i_sample] = 0.0
                    for i_gauss in range(gmm.n_components):
                        mu = gmm.means_[i_gauss]
                        sigma = np.sqrt(gmm.covariances_[i_gauss])
                        x = delta_norm_log[i_sample]
                        hp = hiprec_erf(float(( (x - mu) / (np.sqrt(2)*sigma) )))
                        probs_gmm[i_sample] = probs_gmm[i_sample] + gmm.weights_[i_gauss]*0.5*(1.0 - hp)
                        #if probs_gmm[i_sample] == 0.0:
                        #    print('i_gauss = ',i_gauss,' mu = ',mu, ' sigma = ', sigma,'probs_gmm = ',probs_gmm[i_sample], ' hp = ',hp, ' delta = ', float(( (x - mu) / (np.sqrt(2)*sigma) )))
                    if probs_gmm[i_sample] == 0.0:
                        print('WARNING: zero probability sample  = ',probs_gmm[i_sample])
                        probs_gmm[i_sample] = np.finfo(probs_gmm.dtype).min
        probs[np.logical_not(np.isfinite(probs))] = 1.0 # in this way it gives no contribution to the logarithm
        probs[np.isnan(probs)] = 1.0 # in this way it gives no contribution to the logarithm
        probs[probs < settings.numerical_precision] = settings.numerical_precision
        probs_gmm[np.logical_not(np.isfinite(probs_gmm))] = 1.0 # in this way it gives no contribution to the logarithm
        probs_gmm[np.isnan(probs_gmm)] = 1.0 # in this way it gives no contribution to the logarithm
        min_finite_prob = np.min(probs_gmm[probs_gmm > 0])
        pos_probs_ord = np.sort(probs_gmm[probs_gmm > 0])
        print('Max gap between probabilities: ',np.max(pos_probs_ord[1:] / pos_probs_ord[:-1]))
        print('Number of samples with zero prob = ',np.sum(probs_gmm < min_finite_prob))
        probs_gmm[probs_gmm < min_finite_prob] = min_finite_prob /  (2*np.max(pos_probs_ord[1:] / pos_probs_ord[:-1]))
        #energies = -1.0*np.log(probs_gmm)
        energies = -1.0*np.log(probs)
        if pdf is not None:
            f = plt.figure()
            ax1 = f.add_subplot(111)
            ax1.scatter(rho_norm_log, delta_norm_log, c = energies)
            plt.xlabel('rho_norm_log')
            plt.ylabel('delta_norm_log')
            pdf.savefig()
            plt.close()
        energies.sort()
        if isinstance(ns_clusters,int) or isinstance(ns_clusters,np.int64) : # if a number of clusters was chosen, return the ones in lowest probability regions
            self.ind_cluster_centers = np.argsort(probs_gmm)[0:ns_clusters]
        elif n_stds_delta is not None: # if n_stds_delta was defined return the ones above it
            self.ind_cluster_centers = ind_cluster_centers
        else: # in this case a list of clusters was provided, return the energies for all of them
            delta_energies = np.empty(len(ns_clusters))
            cost = ((energies[1:] - energies[:-1])[-1::-1])
            for i, n_clusters in enumerate(ns_clusters):
                delta_energies[i] = cost[i+1]
                #delta_energies[i] = np.cumsum((energies[1:] - energies[:-1])[-1::-1])[n_clusters-1]/n_clusters
                #delta_energies[i] = ( np.mean(energies[-n_clusters:]) - np.mean(energies[-2*n_clusters:-n_clusters]) )
            return delta_energies
        delta_energy = ( np.mean(energies[-len(self.ind_cluster_centers):]) - np.mean(energies[-2*len(self.ind_cluster_centers):-len(self.ind_cluster_centers)]) )  # this is equal to (energy cluster centers) - (energy other samples above average)
        return delta_energy
    def test_metrics(self, percents, n_stds_delta, ns_clusters = None, pdf = None):
        """
        For parameter definition see DensityPeaks.fit
        """
        delta_energy_best = -np.inf
        for metric in ['euclidean', 'manhattan', 'chebyshev', 'angular']:
            self.dist = self.distance_matrix(metric)
            percent, n_cluster, delta_energy, dummy = self.test_percents_clusters(percents, n_stds_delta, ns_clusters, pdf)
            print('Best energy for metric {0:s} {1:f}'.format(metric, delta_energy))
            if delta_energy > delta_energy_best:
                delta_energy_best = delta_energy
                metric_best = metric
        return metric_best
    def test_percents_clusters(self, percents, n_stds_delta, ns_clusters, pdf):
        """
        Try all the percent values and return the one giving the higher energy

        For parameter definition see DensityPeaks.fit

        Return
        ------
        float
            The percent value giving the higher energy
        float
            The corresponding  energy
        """
        if (n_stds_delta is not None) and (ns_clusters is not None):
            raise ValueError('ERROR: define ns_clusters OR n_stds_delta')
        if isinstance(percents, list):
            if len(percents) == 1:
                percents = percents[0]
        if (isinstance(percents, float) or isinstance(percents, int)) and (isinstance(ns_clusters, float) or isinstance(ns_clusters, int) or isinstance(ns_clusters, np.int64)): # dummy case, both percents and ns_clusters are defined
            return percents, ns_clusters, 0.0, 0.0
        if (isinstance(percents, float) or isinstance(percents, int)):
            percents = [percents,]
        if (ns_clusters is None) or isinstance(ns_clusters,int): # in this case different percents values are tested, clusters is fixed (if defined) or calculated from n_stds_delta
            delta_energies = -np.inf*np.ones(len(percents))
            number_clusters = np.zeros(len(percents))
            for i, percent in enumerate(percents):
                delta_energies[i] = self.get_energy(percent, n_stds_delta, ns_clusters, pdf)
                number_clusters[i] = len(self.ind_cluster_centers)
                print('Test with percent {0:f} - energy {1:f} - n_clusters {2:d}'.format(float(percent), delta_energies[i], int(number_clusters[i])))
            if pdf is not None:
                f = plt.figure()
                ax1 = f.add_subplot(211)
                ax1.plot(percents, delta_energies, 'o-k')
                plt.xlabel('Percentile distance')
                plt.ylabel('Delta energy')
                ax2 = f.add_subplot(212)
                ax2.plot(percents, number_clusters, 'o-k')
                plt.xlabel('Percentile distance')
                plt.ylabel('Number of clusters')
                pdf.savefig()
                plt.xscale('log')
                plt.sca(ax1)
                plt.xscale('log')
                pdf.savefig()
                plt.sca(ax2)
                plt.yscale('log')
                pdf.savefig()
                plt.close()
            delta_energies[np.isnan(delta_energies)] = -np.inf
            ind_best = np.argsort(delta_energies)[-1]
            return percents[ind_best], number_clusters[ind_best], delta_energies[ind_best], delta_energies
        else: # in this case it means that a list of clusters was provided
            delta_energies = -np.inf*np.ones((len(percents), len(ns_clusters)))
            for i, percent in enumerate(percents):
                delta_energies[i,:] = self.get_energy(percent, n_stds_delta, ns_clusters, pdf)
                i_best_cluster = np.argmax(delta_energies[i,:])
                print('Test with percent {0:f} - best n_clusters {1:d} - energy {2:f}'.format(float(percent), int(ns_clusters[i_best_cluster]), delta_energies[i, i_best_cluster]))
            if pdf is not None:
                for i_percent, percent in enumerate(percents):
                    f = plt.figure()
                    ax = f.add_subplot(111)
                    ax.plot(ns_clusters, delta_energies[i_percent,:], 'o-k')
                    if self.verbose > 1:
                        print('Number of clusters = ',ns_clusters)
                        print('Energies = ',delta_energies[i_percent,:])
                    plt.xlabel('Number of clusters')
                    plt.ylabel('Cost Function')
                    plt.title('Percent = {0:f}'.format(float(percent)))
                    pdf.savefig()
                    plt.close()
            if len(percents) > 1:
                f = plt.figure()
                ax = f.add_subplot(211)
                cax = ax.imshow(delta_energies, aspect = 'auto',cmap = 'inferno')
                plt.ylabel('Percentile')
                inds_ticks = np.arange(0,len(ns_clusters),max(1,int(np.round(len(ns_clusters)/10)))).astype(int)
                plt.gca().set_xticks(inds_ticks)
                plt.gca().set_xticklabels(np.array(ns_clusters)[inds_ticks])
                inds_ticks = np.arange(0,len(percents),max(1,int(np.round(len(percents)/10)))).astype(int)
                plt.gca().set_yticks(inds_ticks)
                plt.gca().set_yticklabels(np.array(percents)[inds_ticks])
                f.colorbar(cax)
                ax = f.add_subplot(212)
                cax = ax.imshow(delta_energies / np.max(delta_energies, axis = 1).reshape(delta_energies.shape[0],1), aspect = 'auto', cmap = 'inferno')
                plt.xlabel('Number of clusters')
                plt.ylabel('Percentile')
                inds_ticks = np.arange(0,len(ns_clusters),max(1,int(np.round(len(ns_clusters)/10)))).astype(int)
                plt.gca().set_xticks(inds_ticks)
                plt.gca().set_xticklabels(np.array(ns_clusters)[inds_ticks])
                inds_ticks = np.arange(0,len(percents),max(1,int(np.round(len(percents)/10)))).astype(int)
                plt.gca().set_yticks(inds_ticks)
                plt.gca().set_yticklabels(np.array(percents)[inds_ticks])
                f.colorbar(cax)
                if pdf is not None:
                	pdf.savefig()
                	plt.close()
                else:
                	plt.show()
            #from scipy.signal import argrelextrema
            #argrelextrema(x, np.greater)
            i_best, j_best = np.unravel_index(np.argsort(delta_energies.flatten())[-1],delta_energies.shape)
            return percents[i_best], ns_clusters[j_best], delta_energies[i_best, j_best], delta_energies
    def search_cluster_centers(self, percents = 10.0, metric = 'euclidean', n_stds_delta = None, ns_clusters = None, manual_refine = False, pdf = None, **kwargs):
        """
        Test parameters and find the cluster centers
        """
        if metric == 'test':
            metric = self.test_metrics(percents, n_stds_delta, ns_clusters, pdf)
            print('Proceeding with metric {0:s}'.format(metric))
        self.dist = self.distance_matrix(metric)
        percent, n_clusters, delta_energy, delta_energies = self.test_percents_clusters(percents, n_stds_delta, ns_clusters, pdf)
        if self.verbose > 0:
            print('Clustering with the Distance-Peaks algorithm, with percent {0:f}'.format(percent))
        self.get_energy(percent, n_stds_delta, n_clusters, pdf)
        #--- Manual refine clusters
        if manual_refine:
            rho_norm = (self.rho - np.min(self.rho)) / (np.max(self.rho) - np.min(self.rho))
            delta_norm = (self.delta - np.min(self.delta)) / (np.max(self.delta) - np.min(self.delta))
            self.manual_clusters_selector = graphics.PointInteractor(np.log10(rho_norm), np.log10(delta_norm), self.ind_cluster_centers)
            self.manual_clusters_selector.run()
        return delta_energies
    def fit(self, halo = 0.0,  pdf = None, **kwargs):
        """
        Parameters
        ----------
        percents: float / list
            If float, this is the percentile of the distances used to define the radius of the kernel
            If list, all the values are tested and the one providing the highest energy is choosen
        metric: str
            Metric used to calculate the matrix of pairwise distances
            Possible choices:
                euclidean
                manhattan
                chebyshev
                angular
                test
            If test, all possible metrics are tested and the one providing the highest energy is choosen
        n_stds_delta: float
            If n_clusters is None, the centers of the clusters are the samples that deviates from the average delta(rho) by
            this number standard deviations
        n_clusters: None / int
            If None, the number of clusters is defined automatically
            If int, this is the number of clusters
        halo:   float
            Samples are considered in cluster cores if density is higher than halo*<density_border>
            where <density_border> is the average density of the samples that are closer than a kernel_radius
            from samples of other clusters
            If halo == 1.0, all samples are in core
        manual_refine:  bool
            If True, draw the rho/delta plot to allow a manual refine of the cluster centers
        """
        if len(self.ind_cluster_centers) == 0:
            raise ValueError('ERROR: first run search_cluster_centers')
        if self.manual_clusters_selector is not None:
            self.ind_cluster_centers = self.manual_clusters_selector.get_actives()
            print('Manually selected clusters {0:d}'.format(len(self.ind_cluster_centers)))
        #--- Select clusters
        if self.clusters_analogic.size == 0: # This is needed when running multiple fit changing the halo
            n_discovered_clusters = 0
            for i in range(self.n_frames()):
                if i in self.ind_cluster_centers:
                    self.dtrajs_merged[i] = n_discovered_clusters
                    self.clusters_analogic = np.vstack((self.clusters_analogic.reshape(n_discovered_clusters,self.n_dims()),self.trajs_merged[i,:]))
                    n_discovered_clusters += 1
        if pdf is not None:
            rho_norm = (self.rho - np.min(self.rho)) / (np.max(self.rho) - np.min(self.rho))
            delta_norm = (self.delta - np.min(self.delta)) / (np.max(self.delta) - np.min(self.delta))
            f_all = plt.figure()
            ax1_all = f_all.add_subplot(1,1,1)
            ax1_all.plot(rho_norm,delta_norm,'.k', markersize = 1.0)
            ax1_all.plot(rho_norm[self.ind_cluster_centers],delta_norm[self.ind_cluster_centers],'kx', markersize = 4.0)
            if self.labels_merged is not None:
                for label in range(self.n_labels()):
                    f = plt.figure()
                    ax1 = f.add_subplot(1,1,1)
                    inds = (self.labels_merged == label)
                    ax1.plot(rho_norm,delta_norm,'.k', markersize = 1.0)
                    ax1.plot(rho_norm[self.ind_cluster_centers],delta_norm[self.ind_cluster_centers],'kx', markersize = 4.0)
                    ax1.plot(rho_norm[inds],delta_norm[inds],'o',color=settings.colors[label%len(settings.colors)], markersize = 2.0)
                    ax1_all.plot(rho_norm[inds],delta_norm[inds],'o',color=settings.colors[label%len(settings.colors)], markersize = 2.0)
                    plt.xlabel('rho_norm')
                    plt.ylabel('delta_norm')
                    plt.title('label {0:d} ({1:d} samples)'.format(label, np.sum(inds)))
                    plt.xscale('log')
                    plt.yscale('log')
                    pdf.savefig(f)
                    plt.close(f)
            plt.xscale('log')
            plt.yscale('log')
            pdf.savefig(f_all)
            plt.close(f_all)
        #---- Assigning points to clusters
        ordrho = np.argsort(self.rho)[-1::-1]
        for i in range(self.n_frames()):
            if self.dtrajs_merged[ordrho[i]] == -1:
                self.dtrajs_merged[ordrho[i]] = self.dtrajs_merged[self.nneigh[ordrho[i]]]
        #---- Assigning points to halo
        if self.verbose > 0:
            print('Assigning samples to {0:d} clusters with halo {1:f}'.format(self.n_clusters(),halo))
        mask = np.logical_not(np.eye(self.dist.shape[0]).astype('bool'))
        for i_cluster in range(self.n_clusters()):
            bnd_rhos = []
            i_samples_cluster_bool = (self.dtrajs_merged == i_cluster) # True if sample of cluster i_cluster
            i_others = np.logical_not(i_samples_cluster_bool) # True for samples not in the cluster
            i_samples_cluster = np.where(i_samples_cluster_bool)[0]
            for i_sample in i_samples_cluster: # cycle over all the samples of the cluster
                i_close_others = np.where(self.dist[i_sample,i_others] < self.kernel_radius)[0] # indexes of elements that belongs to this cluster and that are closer than kernel_radius from samples of other clusters
                if len(i_close_others): # there are samples closer than kernel distance belonging to other clusters
                    bnd_rhos.append(self.rho[i_sample])
            if len(bnd_rhos):
                bnd_rhos = np.array(bnd_rhos)
                i_samples_outside_border = i_samples_cluster[self.rho[i_samples_cluster_bool] < halo*np.mean(bnd_rhos)]
                self.dtrajs_merged[i_samples_outside_border] = -1
    def predict(self, traj):
        for i_traj_internal, traj_internal in enumerate(self.trajs): # search for traj among self.trajs
            if self.n_frames_traj(traj_internal) == len(traj):
                if np.all(np.prod(traj_internal == traj, axis = 1).astype(bool)): # this is the internal trajectory
                    i_start, i_end = self.get_index_merged(i_traj_internal)
                    return np.array(self.dtrajs_merged[i_start:i_end])
        raise ValueError('ERROR: missing trajectory')

#class RegularGrid(Cluster):
#    """
#    Class for clustering on a regular grid
#
#    Attributes
#    ----------
#    mins : list
#    maxs : list
#    numbins : list
#    deltas : list
#
#    """
#    def __init__(self, mins, maxs, numbins, **kargs):
#        super(RegularGrid,self).__init__(**kargs)
#        self.mins = mins # Minimum values of the grid alongs all the dimensions
#        self.maxs = maxs # Maximum values of the grid alongs all the dimensions
#        self.numbins = numbins  # Number of bins of the grid alongs all the dimensions
#        self.deltas = []
#        for idim in range(self.n_dims()):
#            self.deltas.append(1.0*(self.maxs[idim] - self.mins[idim])/self.numbins[idim])
#        ibins = []
#        ngrid = np.ones(tuple(self.numbins))
#        for inds, dummy in np.ndenumerate(ngrid): # Iterator over indexes
#            ibins.append(inds)
#        self.ibins = np.array(ibins).astype(np.integer)
#        self.bin_centers = self.mins+(self.ibins+0.5)*self.deltas
#    def discretize(self, traj):
#        """
#        Discretize a trajectory
#
#        Each sample is assigned to the closer bin_centers
#        """
#        dist = sp.spatial.distance.cdist(traj, self.bin_centers)
#        dtraj = np.argmin(dist, axis = 1)
#        return np.ma.masked_invalid(dtraj)
#    def calculate_probability(self, shape_grid = 'cylinder', norm_trajs = False):
#        """
#        Plot the probability
#
#        Parameters
#        ----------
#        norm_trajs : bool
#            True = The sample in each trajectory are independent (separate particles)
#                so when normalizing divide by the number of trajectories
#            False = The trajectory are real trajectories (same particles over time)
#                so when normalizing divide by the number of frames
#        """
#        prob = self.n_samples() # this is just the number of samples for each bin
#        if norm_trajs:
#            prob /= self.n_trajs() # in this way it's the average number of samples per bin over time
#        else:
#            prob /= self.n_frames() # in this way it's the average number of samples per bin over time
#        prob /= self.grid.bin_volumes(shape_grid = shape_grid) # so not it's the average number of samples over A^3
#        return prob
#    def indexes_outside_grid(self, traj):
#        below_low_bnds = np.any(traj < self.mins, axis = 1)
#        above_up_bnds = np.any(traj > self.maxs, axis = 1)
#        return below_low_bnds + above_up_bnds
#    def bin_volumes(self, shape_grid):
#        """
#        Parameters
#        ----------
#        shape_grid : str
#            'cylinder' --> Volumes are calculated considering a cylindrical grid with radial coordinate at position 0
#        """
#        if shape_grid == 'cylinder':
#            volumes = np.empty(self.n_bins())
#            for i_bin in range(self.n_bins()):
#                i_rad = self.ibins[i_bin][0]
#                rad_min = self.mins[0] + i_rad*self.deltas[0]
#                rad_max = self.mins[0] + (i_rad+1)*self.deltas[0]
#                volumes[i_bin] = np.pi*(rad_max**2 - rad_min**2)*self.deltas[1]
#        else:
#            raise ValueError('ERROR: unknown grid shape')
#        return volumes
#
#class MultiGrid(Cluster):
#    """
#    Class for clustering on a regular grid
#
#    Attributes
#    ----------
#    mins : list
#    maxs : list
#    numbins : list
#    deltas : list
#    """
#    def __init__(self, numbins, mins = None, maxs = None, **kargs):
#        """
#        Parameters
#        ----------
#        numbins: int
#            The number of bins along each direction
#        mins: None/list
#            value: float
#                The minimum of the grid along each dimension
#            If None, values are calculated using traj
#        maxs: None/list
#            value: float
#                The maximum of the grid along each dimension
#            If None, values are calculated using traj
#        """
#        super(MultiGrid, self).__init__(**kargs)
#        self.numbins = numbins
#        if mins is None:
#            mins = []
#            for i in range(self.n_dims()):
#                mins.append( min( [np.min(self.trajs[j][:,i]) for j in range(self.n_trajs_analogic())] ) )
#        self.mins = np.array(mins)
#        if maxs is None:
#            maxs = []
#            for i in range(self.n_dims()):
#                maxs.append( max( [np.max(self.trajs[j][:,i]) for j in range(self.n_trajs_analogic())] ) )
#        self.maxs = np.array(maxs)
#        self.deltas = 1.0*(self.maxs - self.mins) / self.numbins
#        self.bin_centers =  self.mins.reshape((1,self.n_dims())) + self.deltas.reshape((1,self.n_dims())) * ( 0.5+np.arange(self.numbins).reshape((self.numbins,1)) ) 
#    def discretize(self, traj):
#        """Discretize a trajectory: each sample is assigned to the closer bin_centers along the corresponding dimension"""
#        dist = np.abs(traj.reshape((self.n_frames_traj(traj),self.n_dims(),1)) - self.bin_centers.transpose())
#        dtraj = np.argmin(dist, axis = 2)
#        return dtraj
#    def analogic(self, dtraj):
#        """Return the discretized trajectory dtraj in analogic form"""
#        traj = np.nan*np.ones((len(dtraj),self.n_dims()))
#        for i in range(self.n_dims()):
#            bin_center = self.bin_centers[:,i]
#            traj[:,i] = bin_center[dtraj[:,i]]
#        return traj
#    def get_minima(self):
#        """
#        Return the bin_center with less points (>0) for each dimension
#        """
#        min_sample_bins = []
#        for i_dim in range(self.n_dims()):
#            n_samples = np.zeros(self.numbins)
#            for i_bin in range(self.numbins):
#                for dtraj in self.dtrajs:
#                    n_samples[i_bin] += np.sum(dtraj[:,i_dim] == i_bin)
#            min_sample_bins.append(self.bin_centers[np.argmin(n_samples),i_dim])
#        return min_sample_bins
#    def get_mapping(self, other):
#        """
#        Calculate the probability to be into a bin of self given each bin of other
#
#        Parameters
#        ----------
#        other: Cluster
#
#        Return
#        ------
#        np.ndarray: <number of dimensions of self> x <number of cluster of self> x <number of cluster of other>
#            [i,j,k] = Probability that dimension i of self belongs to cluster j given that other belongs to cluster k
#        """
#        if self.n_trajs() != other.n_trajs():
#            raise ValueError('ERROR: different number of trajectories between self and other')
#        for i_traj in range(self.n_trajs()):
#            if self.n_frames_traj(self.trajs[i_traj]) != self.n_frames_traj(other.trajs[i_traj]):
#                raise ValueError('ERROR: different number of frames between self and other')
#        mapping = np.zeros((self.n_dims(), self.numbins, other.n_clusters())) 
#        for i_traj in range(self.n_trajs()):
#            for i_bin_other in range(other.n_clusters()):
#                i_frames = (other.dtrajs[i_traj] == i_bin_other)
#                selected_frames_self = self.dtrajs[i_traj][i_frames,:]
#                for i_bin_self in range(self.numbins):
#                    mapping[:,i_bin_self,i_bin_other] += np.sum(selected_frames_self == i_bin_self, axis = 0)
#        mapping /= np.sum(mapping, axis = 1).reshape(self.n_dims(),1,other.n_clusters())
#        mapping[np.isnan(mapping)] = 0.0
#        if np.sum(np.isnan(mapping)):
#            print 'ERROR: wrong mapping'
#            exit()
#        return mapping

class TrainingKNN(ClusterSKlearn):
    def __init__(self, trajs = None, labels = [], verbose = 0):
        super(TrainingKNN, self).__init__(trajs, labels, verbose)
        self.name = 'DP + KNN'
    def fit(self, n_clusters, percent = 10.0, training_samples = 10000, n_rounds = 100, min_n_samples_robust = 4, n_samples_per_cluster = 20, n_neighbors = 3, pdf = None):
        """
        training_samples:   int
            Number of samples used for DensityPeaks
        min_n_samples_robuts    int
            The classification of a sample is considered robust is the sample was classified
            into the same cluster for more than min_n_sample_robust times
        n_samples_per_cluster   int
            Number of samples of each cluster kept in the old set
        """
        from sklearn.neighbors import KNeighborsClassifier
        #--- check paramters
        if (not (isinstance(n_clusters, int) or isinstance(n_clusters,float))) or (not (isinstance(percent, int) or isinstance(percent,float))):
            raise ValueError('ERROR: wrong paramters for TrainingKNN')
        #--- if number of samples is lower than training samples, run density peaks
        if training_samples >= self.n_frames():
            print('Warning: number of samples is lower than training_samples, running DensityPeaks')
            cluster_method = DensityPeaks(self.trajs, self.labels, self.verbose)
            cluster_method.search_cluster_centers(percents = percent, ns_clusters = n_clusters, pdf = pdf)
            cluster_method.fit()
            self.fit_done = True
            self.dtrajs_merged = cluster_method.dtrajs_merged
            return
        #--- create the 1st training set
        clusters_training = -1*np.ones(self.n_frames()) # labels of the training set
        counts_training = np.zeros(self.n_frames()) # How many times a sample was in the training set
        ind_selected_samples_old = np.random.choice(self.n_frames(), training_samples, replace=False) # these samples will be used for clustering with DP
        data_training_old = self.trajs_merged[ind_selected_samples_old,:] # data for samples used for clustering with DP
        #--- run clustering for the 1st training set
        training = DensityPeaks(trajs = [data_training_old], verbose = self.verbose)
        training.search_cluster_centers(percents = percent, metric = 'euclidean', ns_clusters = n_clusters, n_stds_delta = None, manual_refine = False)
        training.fit_predict()
        #--- update the training set
        clusters_old = training.dtrajs_merged
        clusters_training[ind_selected_samples_old] = clusters_old
        counts_training[ind_selected_samples_old] += 1
        for i_round in range(1,n_rounds):
            #--- create the n-th training set
            ind_selected_samples_new = np.random.choice(self.n_frames(), training_samples, replace=False)
            counts_training[ind_selected_samples_old] += 1
            counts_training[ind_selected_samples_new] += 1
            data_training_new = self.trajs_merged[ind_selected_samples_new,:]
            #--- run clustering for the 1st training set
            training = DensityPeaks(trajs = [data_training_old, data_training_new], verbose = self.verbose)
            training.search_cluster_centers(percents = percent, metric = 'euclidean', ns_clusters = n_clusters, n_stds_delta = None, manual_refine = False)
            training.fit_predict()
            #--- update the global training set
            clusters_new = training.dtrajs_merged
            #--- check if DensityPeaks worked properly, otherwise skip this round
            if -1 in clusters_new:
                continue
            #--- couple new clusters with previous ones
            count_matrix = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                ind_i_1 = np.where(clusters_old == i)[0]
                for j in range(n_clusters):
                    count_matrix[i,j] = np.sum(clusters_new[ind_i_1] != j)
            inds_1, inds_2 = linear_sum_assignment(count_matrix)
            inds_new_2_old = list(inds_2)
            for i, ind in enumerate(ind_selected_samples_old):
                if clusters_training[ind] != inds_new_2_old.index(clusters_new[i]): # if they belong to different clusters decrease the counter
                    #print 'Removing {0:d} from training'.format(ind)
                    #clusters_training[ind] = max(clusters_training[ind]-2,-1)
                    counts_training[ind] = max(counts_training[ind]-2,0)
            for i, ind in enumerate(ind_selected_samples_new):
                if clusters_training[ind] == -1: # it's the first time this sample was assigned to a cluster
                    clusters_training[ind] = inds_new_2_old.index(clusters_new[i+len(ind_selected_samples_old)])
                elif clusters_training[ind] > -1:
                    if clusters_training[ind] !=  inds_new_2_old.index(clusters_new[i+len(ind_selected_samples_old)]): # if they belong to different clusters decrease the counter
                        #clusters_training[ind] = max(clusters_training[ind]-2,-1)
                        counts_training[ind] = max(counts_training[ind]-2,0)
            inds_training = np.where(clusters_training >= 0)[0]
            if self.verbose > 0:
                print('Number of samples with some coherent classification = ',len(inds_training))
            inds_training_robust = inds_training[np.where(counts_training[inds_training] > min_n_samples_robust)[0]]
            if self.verbose > 0:
                print('Number of samples with robust classification = ',len(inds_training_robust))
            indexes, counts = np.unique(clusters_training[inds_training_robust], return_counts = True)
            if self.verbose > 0:
                print('Number of samples with robust classification per cluster = ',counts)
            ind_selected_samples_old = []
            for i_cluster in range(n_clusters):
                #--- first add some robust samples
                indexes_cluster = np.where(clusters_training[inds_training_robust] == int(i_cluster))[0]
                if len(indexes_cluster) > int(0.5*n_samples_per_cluster):
                    indexes_cluster_random_selected = list(inds_training_robust[np.random.choice(indexes_cluster, int(0.5*n_samples_per_cluster), replace=False)])
                    if self.verbose > 0:
                        print('Adding {0:d} robust samples'.format(int(0.5*n_samples_per_cluster)))
                else:
                    if self.verbose > 0:
                        print('Adding {0:d} robust samples'.format(len(inds_training_robust[indexes_cluster])))
                    indexes_cluster_random_selected = list(inds_training_robust[indexes_cluster])
                #--- then add other samples
                n_samples_missing = (n_samples_per_cluster - len(indexes_cluster_random_selected))
                if n_samples_missing:
                    if self.verbose > 0:
                        print('Still need to add {0:d} samples'.format(n_samples_missing))
                    indexes_cluster = inds_training[np.where((clusters_training[inds_training] == int(i_cluster)) & (counts_training[inds_training] < min_n_samples_robust))[0]]
                    if self.verbose > 0:
                        print('Samples available: ',len(indexes_cluster))
                    if len(indexes_cluster) > n_samples_missing:
                        inds = indexes_cluster[np.argsort(counts_training[indexes_cluster])[-n_samples_missing:]]
                        indexes_cluster_random_selected.extend(inds)
                        #indexes_cluster_random_selected.extend(np.random.choice(indexes_cluster, n_samples_missing, replace=False))
                    else:
                        indexes_cluster_random_selected.extend(indexes_cluster)
                ind_selected_samples_old.extend(indexes_cluster_random_selected)
            data_training_old = self.trajs_merged[ind_selected_samples_old,:]
            clusters_old = clusters_training[ind_selected_samples_old]
            if (len(set(clusters_training[inds_training_robust].astype(int))) == n_clusters):
                if len(inds_training_robust) < 100:
                    if self.verbose > 0:
                        print('Too few training samples: ',len(inds_training_robust))
                elif np.min(counts) < 5*n_neighbors:
                    if self.verbose > 0:
                        print('Samples are still missing, trying another round ', counts)
                else:
                    inds_training = inds_training_robust
                    break
            else:
                print('Clusters are still missing, trying another round', counts)
        else:
            if n_rounds > 1:
                raise ValueError('ERROR: increase the maximum number of iterations')
        data_training = self.trajs_merged[inds_training_robust,:]
        clusters_training = clusters_training[inds_training_robust].astype(int)
        #print('data_training = ',data_training)
        #print('clusters_training = ',clusters_training)
        #f = plt.figure()
        #ax = f.add_subplot(111)
        #ax.scatter(data_training[:,0],data_training[:,1], c = clusters_training)
        #plt.show()
        #--- Clustering with KNN
        print('Clustering data with KNN algorithm')
        clustering = KNeighborsClassifier(n_neighbors = n_neighbors)
        clustering.fit(data_training, clusters_training)
        self.dtrajs_merged = clustering.predict(self.trajs_merged)
        #print('self.dtrajs_merged = ',self.dtrajs_merged)
        #print('self.dtrajs_merged = ',self.dtrajs_merged.shape)
        #self.bin_centers = np.empty((len(set(dtraj)),self.n_dims()))
        #for i_cluster in range(len(set(dtraj))):
        #    self.bin_centers[i_cluster,:] = np.mean(self.trajs[0][dtraj == i_cluster,:],axis = 0)
        #self.dtrajs = [dtraj]
        self.fit_done = True
    def predict(self, traj):
        for i_traj_internal, traj_internal in enumerate(self.trajs): # search for traj among self.trajs
            if self.n_frames_traj(traj_internal) == len(traj):
                if np.all(np.prod(traj_internal == traj, axis = 1).astype(bool)): # this is the internal trajectory
                    i_start, i_end = self.get_index_merged(i_traj_internal)
                    return np.array(self.dtrajs_merged[i_start:i_end])
        raise ValueError('ERROR: missing trajectory')


def make_data(kind, n_samples = 1000, n_samples_rare = 10):
    """
    Generate toy data sets for testing clustering algorithm

    Parameters
    ----------
    n_samples:  int
        Number of samples per class
    n_samples_rare:  int
        Number of samples per rare classes
    """
    from sklearn import datasets
    if kind == 'circles':
        X,y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif kind == 'moons':
        X,y = datasets.make_moons(n_samples=n_samples, noise=.05)
    elif kind == 'moons+':
        X,y = datasets.make_moons(n_samples=2*n_samples, noise=.05)
        x = np.array([-1.0,-1.0]) + 0.1*np.random.randn(n_samples_rare,2)
        X = np.vstack((X,x))
        y = np.hstack((y,2*np.ones(n_samples_rare)))
        x = np.array([+2.0,1.0]) + 0.1*np.random.randn(n_samples_rare,2)
        X = np.vstack((X,x))
        y = np.hstack((y,3*np.ones(n_samples_rare)))
    elif kind == 'blobs':
        X,y = datasets.make_blobs(n_samples = n_samples, centers = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], cluster_std = [0.25, 0.25, 0.25])
    elif kind == 'gates':
        X = np.array([0.75,1.0]) + 0.15*np.random.randn(n_samples,2)
        y = np.zeros(n_samples)
        x = np.array([1.35,1.0]) + 0.075*np.random.randn(n_samples_rare,2)
        X = np.vstack((X,x))
        y = np.hstack((y, 1*np.ones(n_samples_rare)))
        x = np.array([1.85,1.0]) + 0.15*np.random.randn(n_samples,2)
        X = np.vstack((X,x))
        y = np.hstack((y, 2*np.ones(n_samples)))
        x = np.array([1.75,1.75]) + 0.15*np.random.randn(n_samples,2)
        X = np.vstack((X,x))
        y = np.hstack((y, 3*np.ones(n_samples)))
        #x = np.hstack((np.random.uniform(low = 0.50, high = 1.0, size = (n_samples_rare,1)),np.random.uniform(low = 0.0, high = 1.0, size = (n_samples_rare,1))))
        #X = np.vstack((X,x))
        #y = np.hstack((y, 2*np.ones(n_samples_rare)))
    else:
        raise ValueError('ERROR: {0:s} kind does not exist'.format(kind))
    return [X,], [y,]

def read_data(file_name, read_labels = False):
    """
    file_name:  str
        Name of the file with the testing data
    """
    with open(file_name,'rt') as fin:
        table = []
        labels = []
        for l in fin.readlines():
            lc = l.strip()
            if lc:
                lf = lc.split()
                if lf:
                    if read_labels:
                        row = [float(value) for value in lf[:-1]]
                        labels.append(int(lf[-1]))
                    else:
                        row = [float(value) for value in lf]
                    table.append(row)
        if read_labels:
            return [np.array(table),], [np.array(labels),]
        else:
            return [np.array(table),]

def run(mode, trajs, **kwargs):
    """
    Parameters
    ----------
    mode: str
    """
    labels = kwargs.pop('labels',[])
    if mode == 'Kmeans':
        C = Kmeans(trajs, labels)
    elif mode == 'DensityPeaks':
        C = DensityPeaks(trajs, labels)
        C.search_cluster_centers(**kwargs)
    elif mode == 'MeanShift':
        C = MeanShift(trajs, labels)
    else:
        raise ValueError('ERROR: mode {0:s} is not implemented'.format(mode))
    C.fit_predict(**kwargs)
    return C

if __name__ == '__main__':
    print('------------------')
    print('Testing cluster.py')
    print('------------------')
    np.set_printoptions(precision = np.inf)

    #mp.dps = 10
    #print(type(erf(4.0)))
    ##print(0.5*(1.0 - erf( (x) / (np.sqrt(2)) ) ))
    ##print(type(sperf(x)))
    ##print('{0:.100f}'.format(sperf(x)))
    ##exit()

    #from scipy.special import erf
    ##from mpmath import *
    #mp.dps = 10
    #print(type(erf(4.0)))
    #exit()

    pdf = PdfPages('./test.pdf')

    #X, y = make_data('blobs', 1000)
    #X, y = read_data('../examples/data/cluster/aggregation.txt', True)
    X, y = make_data('gates', 1000, 100)

    #C = run('Kmeans', trajs = X, labels = y, n_clusters = 4, pdf = pdf)
    #C = run('MeanShift', trajs = X, labels = y, bandwidth = 0.6, pdf = pdf)
    C = run('DensityPeaks', trajs = X, labels = y, ns_clusters = 6, pdf = pdf)
    #C = run('DensityPeaks', trajs = X, labels = y, n_clusters = None, percents = [0.1,1.0,10.0], n_stds_delta = 3.0, metric = 'euclidean', halo = 0.0, manual_refine = True, pdf = pdf)
    #        , percents = [0.1,0.2,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0], n_stds_delta = None, metric = 'euclidean', halo = 0.0, manual_refine = True, pdf = pdf)

    #C = TrainingKNN(trajs = X, labels = y)
    #C.fit_predict(percent = 10.0, n_clusters = 4, training_samples = 10000, n_rounds = 100, min_n_samples_robust = 3, n_samples_per_cluster = 50, n_neighbors = 3, pdf = pdf)
    
    
    #C.score()
    C.show(pdf)
    print(C)

    pdf.close()
