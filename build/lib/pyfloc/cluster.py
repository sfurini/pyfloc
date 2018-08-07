#!/usr/bin/env python

import sys
import pickle

import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

colors = ['blue', 'green', 'red', 'cyan', "magenta", 'yellow', 'indigo', 'silver', 'tomato', 'gold', 'springgreen', 'tan', 'cadetblue', 'aqua', 'khaki', 'indianred', 'brown', 'lime', 'ivory', 'lightsalmon', 'teal']
np.set_printoptions(linewidth = np.inf)

class Cluster(object):
    """
    Class for discretizing trajectories

    Classes inheriting Cluster should define:
        fit : method that initialize the clustering algorithm
        predict: method that take as input an analogic trajectory and return its discretized version

    Attributes
    ----------
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
    name: str
        Name of the method adopted
    """
    def __init__(self, trajs, labels = []):
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
        """
        self.trajs = trajs # analogical trajectories
        self.dtrajs = [] # discrete trajectories
        self.clusters_analogic = np.empty(0) # values used as representative of clusters for analogic conversions
        self.name = 'Undefined' # name of the clustering algorithm
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
        if labels is not None:
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
    def reset_trajectories(self, trajs = []):
        """Change the list of trajectories with trajs and reset dtrajs"""
        self.trajs = trajs
        self.dtrajs = []
    def transform(self):
        """Perform the discretization of all the trajectories trajectories"""
        self.dtrajs = []
        for i_traj, traj in enumerate(self.trajs):
            self.dtrajs.append(self.predict(traj))
    def fit_predict(self, *args, **kwargs):
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
    def score(self):
        """
        Calculate the score of the clustering algorithm by comparing the clustering results with reference labels
        """
        from scipy.optimize import linear_sum_assignment
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
            inv_f = -1.0*np.transpose(count_matrix)
            pair_index_labels, pair_index_clusters = linear_sum_assignment(inv_f)
        else:
            inv_f = -1.0*count_matrix
            pair_index_clusters, pair_index_labels = linear_sum_assignment(inv_f)
        precision = np.zeros(n_labels)
        recall = np.zeros(n_labels)
        f = np.zeros(n_labels)
        for i_cluster, i_label in zip(pair_index_clusters, pair_index_labels):
            precision[i_label] = precision_matrix[i_cluster, i_label]
            recall[i_label] = recall_matrix[i_cluster, i_label]
            f[i_label] = f_matrix[i_cluster, i_label]
        print('Count matrix: ',count_matrix)
        print('Pair index labels: ',pair_index_labels)
        print('Pair index clusters: ',pair_index_clusters)
        print('Precision: ',precision,', average = ',np.mean(precision[pair_index_labels]))
        print('Recall: ',recall,', average = ',np.mean(recall[pair_index_labels]))
        print('f: ',f,', average = ',np.mean(f[pair_index_labels]))
        print('Score clustering algorithm: {0:f} +/- {1:f}'.format(np.mean(f[pair_index_labels]),np.std(f[pair_index_labels])))
        return pair_index_labels, pair_index_clusters, precision[pair_index_labels], recall[pair_index_labels], f[pair_index_labels]
    def centroids(self):
        n_clusters = self.n_clusters()
        clusters_analogic = np.empty((n_clusters, self.n_dims()))
        for i_cluster in range(n_clusters):
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
    def show_n_samples(self, pdf):
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
            pdf.savefig()
            plt.close(f)
        elif self.n_dims() == 2:
            f = plt.figure()
            ax = f.add_subplot(111, projection = '3d')
            ax.scatter(self.clusters_analogic[:,0], self.clusters_analogic[:,1], self.n_samples())
            plt.xlabel('Reaction Coordinate 0')
            plt.ylabel('Reaction Coordiante 1')
            pdf.savefig()
            plt.close(f)
        else:
            raise ValueError('ERROR: not implemented for {0:d} dimensions'.format(self.n_dims()))
    def show(self, pdf, plot_trajs = False, plot_maps = True, stride = 1):
        plt.rcParams['image.cmap'] = 'Paired'
        if (not plot_trajs) and (self.n_dims() == 1):
            print('WARNING: nothing to show for 1D data if not plot_traj is False')
            return
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
                                    inds_cluster = (dtraj == i_cluster)
                                    ax1.plot(traj[inds_cluster,i][::stride],traj[inds_cluster,j][::stride],'.',color=colors[i_cluster%len(colors)])
                                    ax1.plot(self.clusters_analogic[i_cluster,i],self.clusters_analogic[i_cluster,j],'o',color=colors[i_cluster%len(colors)], markeredgecolor = 'k')
                        if self.clusters_analogic.size > 0:
                            for i_cluster in range(self.n_clusters()):
                                ax1.plot(self.clusters_analogic[i_cluster,i],self.clusters_analogic[i_cluster,j],'o',color=colors[i_cluster%len(colors)], markeredgecolor = 'k')
                        if self.labels is not None:
                            ax2.scatter(traj[::stride,i],traj[::stride,j],c=self.labels[i_traj][::stride].astype(int), vmin = 0, vmax = 10)
                    plt.xlabel('Feature {0:d}'.format(i))
                    plt.ylabel('Feature {0:d}'.format(j))
                    pdf.savefig()
                    plt.close()
        if self.labels is not None:
            means_labels = np.empty((self.n_labels(),self.n_dims()))
            for label in range(self.n_labels()):
                f = plt.figure()
                ax1 = f.add_subplot(1,1,1)
                means_labels[label,:] =  self.get_mean_label(label)
                ax1.errorbar(range(self.n_dims()), means_labels[label,:], yerr = self.get_std_label(label), fmt = 'o', label = label)
                plt.xlabel('Feature')
                plt.ylabel('mean +/- std')
                plt.legend()
                plt.ylim([0,1.0])
                pdf.savefig()
                plt.close()
            plt.rcParams['image.cmap'] = 'bwr'
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            cax = ax.matshow(means_labels)
            f.colorbar(cax)
            plt.xlabel('Features')
            plt.ylabel('Labels')
            pdf.savefig()
            plt.close()
    def __str__(self):
        output = 'Results of clustering with method {0:s}\n'.format(self.name)
        n_samples_per_cluster = self.n_samples()
        if self.labels is not None:
            n_samples_per_label = self.n_samples_labels()
            for i, label in enumerate(np.argsort(n_samples_per_label)):
                output += '\tLabel {0:d} = {1:d}\n'.format(int(label), int(n_samples_per_label[label]))
                output += '\t\t means = ' + str(self.get_mean_label(label)) + '\n'
                output += '\t\t stds = ' + str(self.get_std_label(label)) + '\n'
            pairing_labels, pairing_clusters, precision, recall, f = self.score()
        for i_cluster in range(self.n_clusters()):
            output += '\tCluster {0:d}\n'.format(i_cluster)
            output += '\t\tnumber of samples = {0:d}\n'.format(int(n_samples_per_cluster[i_cluster]))
            if self.labels is not None:
                if self.n_labels() >= self.n_clusters():
                    output += '\t\tpairing label[{0:d}] = {1:d}\n'.format(pairing_labels[i_cluster], int(n_samples_per_label[pairing_labels[i_cluster]]))
                    output += '\t\tprecision = {0:f}\n'.format(precision[i_cluster])
                    output += '\t\trecall = {0:f}\n'.format(recall[i_cluster])
                    output += '\t\tf-score = {0:f}\n'.format(f[i_cluster])
                else:
                    if i_cluster in pairing_clusters:
                        i_label = np.where(pairing_clusters == i_cluster)[0][0]
                        label = pairing_labels[i_label]
                        output += '\t\tpairing label[{0:d}] = {1:d}\n'.format(label, int(n_samples_per_label[label]))
                        output += '\t\tprecision = {0:f}\n'.format(precision[i_label])
                        output += '\t\trecall = {0:f}\n'.format(recall[i_label])
                        output += '\t\tf-score = {0:f}\n'.format(f[i_label])
        output += '\tAverage f-score = {0:f} +/- {1:f}\n'.format(np.mean(f), np.std(f))
        return output[:-1]
    def dump(self, file_name):
        with open(file_name,'wb') as fout:
            pickle.dump(self, fout)

class Unique(Cluster):
    """
    Class for clustering each unique point in its own cluster
    N.B. It works only when feature values are integers

    Attributes
    ----------
    list_samples: list
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
    trajs_merged : np.ndarray
        Shape: <number of samples> x <number of features>
        All the trajectories combined into a single array
    labels_merged: np.ndarray
        Shape: <number of samples>
        All the labels combined into a single array
    """
    def __init__(self, trajs = None, labels = None):
        super(ClusterSKlearn, self).__init__(trajs, labels)
    def predict(self, traj):
        return self.algorithm.predict(traj)
    def get_index_merged(self, i_traj):
        """Return start and end index for trajetory i_traj in merged trajectory"""
        ind_start = 0
        ind_end = self.n_frames_traj(self.trajs[0])
        for i_traj in range(1, i_traj):
            ind_start += self.n_frames_traj(self.trajs[i_traj])
            ind_end += self.n_frames_traj(self.trajs[i_traj])
        return ind_start, ind_end

class Kmeans(ClusterSKlearn):
    """
    """
    def __init__(self, trajs = None, labels = None):
        from sklearn.cluster import KMeans
        self.algorithm = KMeans
        super(Kmeans, self).__init__(trajs, labels)
        self.name = 'Kmeans'
    def fit(self, n_clusters, *args, **kwargs):
        print('Clustering data with Kmeans algorithm...')
        self.algorithm = self.algorithm(n_clusters = n_clusters, n_init = 1000, init = 'k-means++', max_iter = 10000)
        self.algorithm.fit(self.trajs_merged)
        self.clusters_analogic = self.algorithm.cluster_centers_

class MiniBatchKmeans(ClusterSKlearn):
    def __init__(self, trajs = None, labels = None):
        from sklearn.cluster import MiniBatchKMeans
        self.algorithm = MiniBatchKMeans
        super(MiniBatchKmeans, self).__init__(trajs, labels)
        self.name = 'MiniBatchKmeans'
    def fit(self, n_clusters, *args, **kwargs):
        print('Clustering data with MiniBatchKmeans algorithm...')
        self.algorithm = self.algorithm(n_clusters = n_clusters, n_init = 10, init = 'k-means++', max_iter = 300, batch_size = 100)
        self.algorithm.fit(self.trajs_merged)
        self.clusters_analogic = self.algorithm.cluster_centers_

class Birch(ClusterSKlearn):
    """
    """
    def __init__(self, trajs = None, labels = None):
        from sklearn.cluster import Birch as BIrch
        self.algorithm = BIrch
        super(Birch, self).__init__(trajs, labels)
        self.name = 'Birch'
    def fit(self, n_clusters, *args, **kwargs):
        """
        threshold: float
            Il raggio dei subcluster deve essere inferiore
        branching_factor: int
            Il numero di figli di un nodo deve essere inferiore
        """
        self.algorithm = self.algorithm(n_clusters = n_clusters, threshold = 0.5, branching_factor = 50, compute_labels = True)
        print('Clustering data with Birch algorithm')
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
        self.algorithm = self.algorithm(damping = 0.5, max_iter = 1000)
        print('Clustering data with AffinityPropagation algorithm')
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
        print('Clustering data with Gaussian Mixture algorithm...')
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

class DensityPeaks(ClusterSKlearn):
    #def compute_rho_delta_grid(self, data, nbins = 100, threshold = 0.01):
    #    kde = sp.stats.gaussian_kde(data)
    #    z = kde(data)
    #    x = np.ma.masked_where(z > threshold, data[0])
    #    y = np.ma.masked_where(z > threshold, data[1])
    #    print(z[x.mask == False])
    #    print(x[x.mask == False])
    #    xedges = np.linspace(0,1,nbins)
    #    yedges = np.linspace(0,1,nbins)
    #    xx,yy = np.meshgrid(xedges, yedges)
    #    gridpoints = np.array([xx.ravel(), yy.ravel()])
    #    zz = np.reshape(kde(gridpoints), xx.shape)
    #    return x, y, xx, yy, zz
    """
    Attributes
    ----------
    dtrajs_merged : np.ndarray
        Shape: <number of samples>
        All the discretized trajectories combined into a single array
    halos_merged : np.ndarray
        Shape: <number of samples>
    """
    def __init__(self, trajs = None, labels = None):
        super(DensityPeaks, self).__init__(trajs, labels)
        self.dtrajs_merged = -1*np.ones(self.n_frames()).astype(int)
        self.name = 'Density Peaks'
    def fit(self, n_clusters, kernel_radius = None, percent = 1.0, pdf = None):
        """
        Parameters
        ----------
        kernel_radius: float
            Radius of the gaussian kernel
        percent: float
            Average percentage of neighbours 
        """
        from sklearn.metrics import pairwise_distances
        def gauss(x, mu, sigma):
            y = (1.0 / (sigma*np.sqrt(2.0*np.pi))) * np.exp(-0.5*np.power((x - mu)/sigma, 2.0))
            return y
        print('Clustering with the Distance-Peaks algorithm, with percent = {0:f}'.format(percent))
        print('Calculating distance matrix...')
        #dist_dims = np.zeros((self.n_dims(),self.n_frames(),self.n_frames()))
        #for i_dim in range(self.n_dims()):
        #    dist_i = pairwise_distances(self.trajs_merged[:,i_dim].reshape(self.n_frames(),-1))
        #    #dist_i = np.digitize(dist_i, bins = np.linspace(0,1,10))-1.0
        #    dist_i[dist_i > 0.25] = 0.25
        #    dist_dims[i_dim,:,:] = dist_i
        #    #dist_i[dist_i < 0.1] = 0.0
        #    #dist_i_sig = 1.0 / (1.0 + np.exp(-10*(dist_i-0.1)))
        #    #dist_i_power = np.power(dist_i/0.25, 4.0)
        #    #dist_i_sig = dist_i_power / (1.0 + dist_i_power)
        #    #dist += dist_i
        #    #print('Label ',label,' ',np.sum(inds))
        #    #print('x ',x[:,i_dim])
        #    #print('min(x) ',np.min(x[:,i_dim]))
        #    #print('max(x) ',np.max(x[:,i_dim]))
        #    #print('mean(x) ',np.mean(x[:,i_dim]))
        #    #print('std(x) ',np.std(x[:,i_dim]))
        #    #print('dist_i_sig ',dist_i_sig)
        #    #f = plt.figure()
        #    #ax1 = f.add_subplot(2,1,1)
        #    #ax2 = f.add_subplot(2,1,2)
        #    #h,e = np.histogram(dist_i, bins = 100)
        #    #b = 0.5*(e[:-1] + e[1:])
        #    #ax1.plot(b,h)
        #    #h,e = np.histogram(dist_i_sig, bins = 100)
        #    #b = 0.5*(e[:-1] + e[1:])
        #    #ax2.plot(b,h)
        #    #pdf.savefig()
        #    #plt.close()
        #dist = np.sqrt(np.sum(np.power(dist_dims,2.0), axis = 0)) # Euclidean
        dist = pairwise_distances(self.trajs_merged) # Euclidean
        #scalar = np.dot(self.trajs_merged, self.trajs_merged.transpose())
        #teta = (scalar / np.sqrt(scalar.diagonal().reshape(scalar.shape[0],1)) / np.sqrt(scalar.diagonal().reshape(1,scalar.shape[0])))
        #teta[teta > 1.0] = 1.0
        #dist = np.arccos(teta) # Angular distance
        #dist = np.sum(dist_dims, axis = 0) # Manahattan
        #dist = np.min(dist_dims, axis = 0) # Chebyshev
        #dist_sort = np.sort(dist_dims, axis = 0)
        #dist = np.sum(dist_sort[:-2,:,:], axis = 0)
        if kernel_radius is None:
            print('Calculating kernel radius...')
            kernel_radius = np.percentile(dist.flatten(), percent)
        #kernel_radius = 2.0
        print('Computing densities with gaussian kernel of radius: {0:12.6f}'.format(kernel_radius))
        #--- Computing rho
        kernel_matrix = np.exp(-np.power(dist/kernel_radius,2.0))
        rho = np.sum(kernel_matrix, axis = 1)
        #--- Computing delta
        ordrho = np.argsort(rho)[-1::-1]
        delta = np.zeros(self.n_frames())
        nneigh = np.zeros(self.n_frames()).astype(int)
        rho_sorted = rho[ordrho]
        #--- Searching neighbors
        print('Searching neighbours...')
        nneigh[ordrho[0]] = np.argmax(dist[ordrho[0]])
        delta[ordrho[0]] = np.max(dist[ordrho[0]])
        for i in range(1,self.n_frames()):
            delta[ordrho[i]] = np.min(dist[ordrho[i]][ordrho[0:i]])
            nneigh[ordrho[i]] = np.where(dist[ordrho[i]]==delta[ordrho[i]])[0][0]    
        #--- Plotting rho-delta relation
        if pdf is not None:
            f = plt.figure()
            ax1 = f.add_subplot(2,1,1)
            ax2 = f.add_subplot(2,1,2)
            ax1.plot(rho,delta,'.k', markersize = 1.0)
            plt.xlabel('rho')
            plt.ylabel('delta')
        #--- Selecting the centers of the clusters
        rho_norm = (rho - np.min(rho)) / (np.max(rho) - np.min(rho))
        delta_norm = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))
        if pdf is not None:
            ax2.plot(rho_norm,delta_norm,'.k', markersize = 1.0)
            plt.xlabel('rho_norm')
            plt.ylabel('delta_norm')
        rho_norm_log = np.log10(rho_norm)
        delta_norm_log = np.log10(delta_norm)
        #--- Sort in log space
        #rho_delta_norm_log = np.vstack((rho_norm_log, delta_norm_log))
        #dist_best = np.linalg.norm(rho_delta_norm_log, axis = 0)
        #ind_cluster_centers = np.argsort(dist_best)[:n_clusters]
        #--- Sort using probabilitie
        inds_finite = np.isfinite(rho_norm_log*delta_norm_log)
        n_bins_rho = 10
        n_bins_delta = 100
        H, xe, ye = np.histogram2d(rho_norm_log[inds_finite], delta_norm_log[inds_finite], bins = [n_bins_rho, n_bins_delta])
        xb = 0.5*(xe[:-1] + xe[1:])
        yb = 0.5*(ye[:-1] + ye[1:])
        outliers = np.inf*np.ones(self.n_frames())
        for ir in range(n_bins_rho):
            inds_sample_bin = (rho_norm_log > xe[ir]) * (rho_norm_log <= xe[ir+1]) * np.isfinite(rho_norm_log) * np.isfinite(delta_norm_log)
            if np.sum(inds_sample_bin) > 20:
                h = H[ir,:] / np.sum(H[ir,:])
                h_gauss = gauss(yb, np.mean(delta_norm_log[inds_sample_bin]), np.std(delta_norm_log[inds_sample_bin]))
                h_gauss /= np.sum(h_gauss)
                outliers[inds_sample_bin] =  gauss(delta_norm_log[inds_sample_bin], np.mean(delta_norm_log[inds_sample_bin]), np.std(delta_norm_log[inds_sample_bin]))
                inds_below_average = inds_sample_bin * (delta_norm_log <  np.mean(delta_norm_log[inds_sample_bin]))
                outliers[inds_below_average] = np.inf
                if pdf is not None:
                    f = plt.figure()
                    ax = f.add_subplot(211)
                    ax.plot(rho_norm_log[inds_sample_bin], delta_norm_log[inds_sample_bin],'.r')
                    ax = f.add_subplot(212)
                    ax.plot(yb, h_gauss, '-k')
                    pdf.savefig()
                    plt.close()
        ind_cluster_centers = np.argsort(outliers)[:n_clusters]
        ##--- Sort along delta axis
        #ind_cluster_centers = np.argsort(delta_norm)[-n_clusters:]
        #--- Select clusters
        n_discovered_clusters = 0
        for i in range(self.n_frames()):
            if i in ind_cluster_centers:
                self.dtrajs_merged[i] = n_discovered_clusters
                self.clusters_analogic = np.vstack((self.clusters_analogic.reshape(n_discovered_clusters,self.n_dims()),self.trajs[0][i,:]))
                n_discovered_clusters += 1
                if pdf is not None:
                    ax1.plot(rho[i],delta[i],'o',color=colors[self.dtrajs_merged[i]%len(colors)], markersize = 1.0)
                    ax2.plot(rho_norm[i],delta_norm[i],'o',color=colors[self.dtrajs_merged[i]%len(colors)], markersize = 1.0)
        #for i in range(self.n_frames()):
        #    if rho_norm[i] > 0.05:
        #        if delta_norm[i] > 0.2:
        #            self.dtrajs_merged[i] = n_discovered_clusters
        #            self.clusters_analogic = np.vstack((self.clusters_analogic.reshape(n_discovered_clusters,self.n_dims()),self.trajs[0][i,:]))
        #            n_discovered_clusters += 1
        #            if pdf is not None:
        #                ax1.plot(rho[i],delta[i],'o',color=colors[self.dtrajs_merged[i]%len(colors)], markersize = 1.0)
        #                ax2.plot(rho_norm[i],delta_norm[i],'o',color=colors[self.dtrajs_merged[i]%len(colors)], markersize = 1.0)
        if pdf is not None:
            #pdf.savefig()
            plt.xscale('log')
            plt.yscale('log')
            pdf.savefig()
            plt.close()
            f_all = plt.figure()
            ax1_all = f_all.add_subplot(1,1,1)
            if self.labels_merged is not None:
                for label in range(self.n_labels()):
                    f = plt.figure()
                    ax1 = f.add_subplot(1,1,1)
                    inds = (self.labels_merged == label)
                    ax1.plot(rho_norm,delta_norm,'.k', markersize = 1.0)
                    ax1.plot(rho_norm[ind_cluster_centers],delta_norm[ind_cluster_centers],'kx', markersize = 4.0)
                    ax1.plot(rho_norm[inds],delta_norm[inds],'o',color=colors[label%len(colors)], markersize = 2.0)
                    ax1_all.plot(rho_norm[inds],delta_norm[inds],'o',color=colors[label%len(colors)], markersize = 2.0)
                    plt.xlabel('rho_norm')
                    plt.ylabel('delta_norm')
                    plt.title('label {0:d} ({1:d} samples)'.format(label, np.sum(inds)))
                    #pdf.savefig()
                    plt.xscale('log')
                    plt.yscale('log')
                    pdf.savefig(f)
                    plt.close(f)
            #pdf.savefig(f_all)
            plt.xscale('log')
            plt.yscale('log')
            pdf.savefig(f_all)
            plt.close(f_all)
        #---- Assigning points to clusters
        for i in range(self.n_frames()):
            if self.dtrajs_merged[ordrho[i]] == -1:
                self.dtrajs_merged[ordrho[i]] = self.dtrajs_merged[nneigh[ordrho[i]]]
        #---- Assigning points to halo
        #self.halos_merged = np.copy(self.dtrajs_merged)
        #if n_discovered_clusters > 1:
        #    bord_rho = np.zeros(n_discovered_clusters)
        #    for i in range(self.n_frames()-1):
        #        for j in range(i+1,self.n_frames()):
        #            if (self.dtrajs_merged[i] != self.dtrajs_merged[j]) and (dist[i,j] <= kernel_radius):
        #                rho_aver = 0.5 * (rho[i] + rho[j])
        #                if rho_aver > bord_rho[self.dtrajs_merged[i]]:
        #                    bord_rho[self.dtrajs_merged[i]] = rho_aver
        #                if rho_aver > bord_rho[self.dtrajs_merged[j]]:
        #                    bord_rho[self.dtrajs_merged[j]] = rho_aver
        #    for i in range(self.n_frames()):
        #        if rho[i] < bord_rho[self.dtrajs_merged[i]]:
        #            self.halos_merged[i] = -1
    def predict(self, traj):
        for i_traj_internal, traj_internal in enumerate(self.trajs): # search for traj among self.trajs
            if self.n_frames_traj(traj_internal) == len(traj):
                if np.all(np.prod(traj_internal == traj, axis = 1).astype(bool)): # this is the internal trajectory
                    i_start, i_end = self.get_index_merged(i_traj_internal)
                    return np.array(self.dtrajs_merged[i_start:i_end])
                    #return np.array(self.halos_merged[i_start:i_end])
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
#
#
#
#
#
#
#class TrainingKNN(Cluster):
#    def __init__(self, training_method, trajs = None, ref_labels = None):
#        trajs = [np.vstack([traj for traj in trajs]).astype('float64')]
#        if ref_labels is not None:
#            ref_labels = [np.hstack([ref_label for ref_label in ref_labels]).astype('int')]
#        self.training_method = training_method
#        super(TrainingKNN,self).__init__(trajs, ref_labels)
#    def run(self, n_clusters, pdf, training_samples = 10, n_rounds = 10, n_neighbors = 5, min_n_samples_robust = 10, n_samples_per_cluster = 10):
#        """
#        min_n_samples_robuts    int
#            The classification of a sample is considered robust is the sample was classified
#            into the same cluster for more than min_n_sample_robust times
#        n_samples_per_cluster   int
#            Number of samples of each cluster kept in the old set
#        """
#        if training_samples > self.n_frames():
#            raise ValueError('ERROR: asking for more training samples than actual samples')
#        from sklearn.neighbors import KNeighborsClassifier
#        #--- create the training set
#        clusters_training = -1*np.ones(self.n_frames())
#        counts_testing = np.zeros(self.n_frames())
#        ind_selected_samples_old = np.random.choice(self.n_frames(), training_samples, replace=False)
#        data_training_old = self.trajs[0][ind_selected_samples_old,:]
#        #training = Agglomerative(n_clusters, trajs = [data_training_old])
#        training = self.training_method(n_clusters, trajs = [data_training_old], percent = 10.0)
#        training.run()
#        clusters_old = training.dtrajs[0]
#        clusters_training[ind_selected_samples_old] = clusters_old
#        counts_testing[ind_selected_samples_old] += 1
#        inds_training = np.where(clusters_training >= 0)[0]
#        for i_round in range(1,n_rounds):
#            ind_selected_samples_new = np.random.choice(self.n_frames(), training_samples, replace=False)
#            counts_testing[ind_selected_samples_old] += 1
#            counts_testing[ind_selected_samples_new] += 1
#            data_training_new = self.trajs[0][ind_selected_samples_new,:]
#            #training = Agglomerative(n_clusters, trajs = [data_training_old, data_training_new])
#            training = self.training_method(n_clusters, trajs = [data_training_old, data_training_new], percent = 10.0)
#            training.run()
#            #training.show(pdf)
#            clusters_new = training.dtrajs[0]
#            count_matrix = np.zeros((n_clusters, n_clusters))
#            for i in range(n_clusters):
#                ind_i_1 = np.where(clusters_old == i)[0]
#                for j in range(n_clusters):
#                    count_matrix[i,j] = np.sum(clusters_new[ind_i_1] != j)
#            inds_1, inds_2 = linear_sum_assignment(count_matrix)
#            inds_new_2_old = list(inds_2)
#            #print 'ind_selected_samples_old = ',ind_selected_samples_old
#            #print 'clusters_old = ',clusters_old
#            #print 'ind_selected_samples_new = ',ind_selected_samples_new
#            #print 'clusters_new = ',clusters_new
#            #print 'inds_1 = ',inds_1
#            #print 'inds_2 = ',inds_2
#            for i, ind in enumerate(ind_selected_samples_old):
#                #print 'clusters_training[ind] = ',clusters_training[ind]
#                #print 'clusters_new[i] = ',clusters_new[i]
#                #print 'inds_new_2_old.index(clusters_new[i]) = ',inds_new_2_old.index(clusters_new[i])
#                if clusters_training[ind] != inds_new_2_old.index(clusters_new[i]):
#                    #print 'Removing {0:d} from training'.format(ind)
#                    clusters_training[ind] = max(clusters_training[ind]-2,-1)
#            for i, ind in enumerate(ind_selected_samples_new):
#                #print 'ind = ',ind
#                #print 'clusters_new[i+len(ind_selected_samples_old)] = ',clusters_new[i+len(ind_selected_samples_old)]
#                #print 'inds_new_2_old.index(clusters_new[i+len(ind_selected_samples_old)]) = ',inds_new_2_old.index(clusters_new[i+len(ind_selected_samples_old)])
#                if clusters_training[ind] == -1:
#                    clusters_training[ind] = inds_new_2_old.index(clusters_new[i+len(ind_selected_samples_old)])
#                elif clusters_training[ind] > -1:
#                    if clusters_training[ind] !=  inds_new_2_old.index(clusters_new[i+len(ind_selected_samples_old)]):
#                        #print 'Removing {0:d} from training'.format(ind)
#                        clusters_training[ind] = max(clusters_training[ind]-2,-1)
#            inds_training = np.where(clusters_training >= 0)[0]
#            print 'Number of samples with some coherent classification = ',len(inds_training)
#            inds_training_robust = inds_training[np.where(counts_testing[inds_training] > min_n_samples_robust)[0]]
#            print 'Number of samples with robust classification = ',len(inds_training_robust)
#            indexes, counts = np.unique(clusters_training[inds_training_robust], return_counts = True)
#            print 'Number of samples with robust classification per cluster = ',counts
#            ind_selected_samples_old = []
#            for i_cluster in range(n_clusters):
#                #--- first add some robust samples
#                indexes_cluster = np.where(clusters_training[inds_training_robust] == int(i_cluster))[0]
#                if len(indexes_cluster) > int(0.5*n_samples_per_cluster):
#                    indexes_cluster_random_selected = list(inds_training_robust[np.random.choice(indexes_cluster, int(0.5*n_samples_per_cluster), replace=False)])
#                    print 'Adding {0:d} robust samples'.format(int(0.5*n_samples_per_cluster))
#                else:
#                    print 'Adding {0:d} robust samples'.format(len(inds_training_robust[indexes_cluster]))
#                    indexes_cluster_random_selected = list(inds_training_robust[indexes_cluster])
#                #--- then add other samples
#                n_samples_missing = (n_samples_per_cluster - len(indexes_cluster_random_selected))
#                if n_samples_missing:
#                    print 'Still need to add {0:d} samples'.format(n_samples_missing)
#                    indexes_cluster = inds_training[np.where((clusters_training[inds_training] == int(i_cluster)) & (counts_testing[inds_training] < min_n_samples_robust))[0]]
#                    print 'Samples available: ',len(indexes_cluster)
#                    if len(indexes_cluster) > n_samples_missing:
#                        inds = indexes_cluster[np.argsort(counts_testing[indexes_cluster])[-n_samples_missing:]]
#                        indexes_cluster_random_selected.extend(inds)
#                        #indexes_cluster_random_selected.extend(np.random.choice(indexes_cluster, n_samples_missing, replace=False))
#                    else:
#                        indexes_cluster_random_selected.extend(indexes_cluster)
#                ind_selected_samples_old.extend(indexes_cluster_random_selected)
#            data_training_old = self.trajs[0][ind_selected_samples_old,:]
#            clusters_old = clusters_training[ind_selected_samples_old]
#            print 'clusters_old: ',clusters_old
#            print 'counts_testing[ind_selected_samples_old]: ',counts_testing[ind_selected_samples_old]
#            if (len(set(clusters_training[inds_training_robust].astype(int))) == n_clusters):
#                if len(inds_training_robust) < 100:
#                    print 'Too few training samples: ',len(inds_training_robust)
#                elif np.min(counts) < n_neighbors:
#                    print 'Samples are still missing, trying another round ', counts
#                else:
#                    inds_training = inds_training_robust
#                    print 'Using training data with at least {0:d} tests'.format(min_n_samples_robust)
#                    print 'Clusters: ',set(clusters_training[inds_training_robust].astype(int))
#                    break
#            else:
#                print 'Clusters are still missing, trying another round', counts
#        else:
#            if n_rounds > 1:
#                raise ValueError('ERROR: increase the maximum number of iterations')
#        data_training = self.trajs[0][inds_training_robust,:]
#        clusters_training = clusters_training[inds_training_robust].astype(int)
#        #--- show the training set
#        training = self.training_method(n_clusters, trajs = [data_training], ref_labels = clusters_training)
#        training.show(pdf)
#        #--- clustering with KNN
#        print 'Clustering data with KNN algorithm'
#        clustering = KNeighborsClassifier(n_neighbors = n_neighbors)
#        clustering.fit(data_training, clusters_training)
#        dtraj = clustering.predict(self.trajs[0])
#        self.bin_centers = np.empty((len(set(dtraj)),self.n_dims()))
#        for i_cluster in range(len(set(dtraj))):
#            self.bin_centers[i_cluster,:] = np.mean(self.trajs[0][dtraj == i_cluster,:],axis = 0)
#        self.dtrajs = [dtraj]
#

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
        X,y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5])
    elif kind == 'gates':
        X = np.array([0.25,0.25]) + 0.1*np.random.randn(n_samples,2)
        y = np.zeros(n_samples)
        x = np.array([0.25,0.75]) + 0.1*np.random.randn(n_samples,2)
        X = np.vstack((X,x))
        y = np.hstack((y, 1*np.ones(n_samples)))
        x = np.hstack((np.random.uniform(low = 0.50, high = 1.0, size = (n_samples_rare,1)),np.random.uniform(low = 0.0, high = 1.0, size = (n_samples_rare,1))))
        X = np.vstack((X,x))
        y = np.hstack((y, 2*np.ones(n_samples_rare)))
    else:
        raise ValueError('ERROR: {0:s} kind does not exist'.format(kind))
    return [X,], [y,]

def run(mode, trajs, **kwargs):
    """
    Parameters
    ----------
    mode: str
    """
    labels = kwargs.pop('labels',None)
    if mode == 'Kmeans':
        C = bamboo.cluster.Kmeans(trajs, labels)
    elif mode == 'DensityPeaks':
        C = bamboo.cluster.DensityPeaks(trajs, labels)
    elif mode == 'MeanShift':
        C = bamboo.cluster.MeanShift(trajs, labels)
    else:
        raise ValueError('ERROR: mode {0:s} is not implemented'.format(mode))
    C.fit_predict(**kwargs)
    return C

if __name__ == '__main__':
    print('------------------')
    print('Testing cluster.py')
    print('------------------')
    pdf = PdfPages('./test_clustering.pdf')

    X, labels = make_data('gates', 2000, 20)
    #C = run('Kmeans', X, labels = labels, n_clusters = 4, pdf = pdf)
    #C = run('MeanShift', X, labels = labels, bandwidth = 0.6, pdf = pdf)
    C = run('DensityPeaks', X, labels = labels, n_clusters = 3, pdf = pdf, percent = 5.0)
    C.score()
    C.show(pdf)
    print(C)
    #for method in [Kmeans, MiniBatchKmeans, Birch, Agglomerative, GaussianMixture, DensityPeaks]:
    #    C = method(trajs = X, labels = labels)
    #    C.fit_predict(n_clusters = 3, pdf = pdf)
    #    C.show(pdf)
    #for method in [AffinityPropagation, MeanShift]:
    #    C = method(trajs = X, labels = labels)
    #    C.fit_predict()
    #    C.show(pdf)

    pdf.close()
