#!/usr/bin/env python
"""
This module proposes a class and methods to slice a 1D timserie (signal) into a set of parts (phases) separated by breakpoints.
"""


__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
import numpy as np

class SignalDecomposer(ABC):
    """Abstract class for signal decomposer module. A SignalDecomposer class provides essentially methods that delivers a list of breakpoints for a given one dimensional signal."""
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def decompose(self):
        """Method to return breakpoints."""
        pass


def get_lowest_cluster(labels, signal):
    """Look for the cluster label having the lowest ordinates values.

    Args:
        labels (ndarray): array of labels associated with each signal value.add()
        signal (ndarray): array of signal values

    Retuns:
        label0: label with the lowest values.
    """
    unique_labels = np.unique(labels).tolist()
    mean_cluster_values = [np.mean(signal[np.where(labels==label, True, False)]) for label in unique_labels]

    return unique_labels[mean_cluster_values.index(np.min(mean_cluster_values))]

class KmeansSignalDecomposer(SignalDecomposer):
    """Implements signal decomposer based on kmeans clustering."""
    def __init__(self, signal):
        # convert any iterable into numpy array of size (n, 1)
        self.signal = np.array(list(signal)).reshape(-1, 1)

    def get_optimal_clustering(self, v0_threshold = 0.05):
        """Get the optimal number of clusters when cluster0 average values are below v0_threshold comparing to total mean values. The clustering is done in the 1-dimension that carries the dataflow signal.

        Args:
            v0_threshold (float, optional): relative mean value of the average lowest level of cluster point ordinates. Defaults to 0.05.

        Returns:
            tuple (int, sklearn.cluster.KMeans): return the optimal number of cluster complying the threshold and the related KMeans object.
        """
        n_clusters = 1
        v0_weight = 1
        while v0_weight > v0_threshold:
            n_clusters += 1
            if n_clusters > len(self.signal):
                # if n_clusters exceeds signal points, exit the loop
                break
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.signal)
            label0 = get_lowest_cluster(kmeans.labels_, self.signal)
            v0_indexes = np.where(kmeans.labels_==label0, True, False)
            v0_weight = self.signal[v0_indexes].sum() / self.signal.sum()


        return n_clusters, kmeans

    def get_breakpoints_and_labels(self, kmeans, merge=False):
        """Returns breakpoints and labels from clustering algorithm. Labels could be merged
        Args:
            kmeans (sklearn.cluster.KMeans): clustering object
            merge (bool, optional): if True labels > 0 will be merged into one label. Defaults to False.

        Returns:
            (breakpoints, labels): list of breakpoints where each element is the index that separates two phases. Each phase get assigned a label to identify its nature later.
        """
        ab = np.arange(len(self.signal))
        labels = np.where(kmeans.labels_ > 0, 1, 0) if merge else kmeans.labels_
        return  ab[np.insert(np.where(np.diff(labels)!=0, True, False), 0, False)].tolist(), labels


    def decompose(self):
        """Common method without proper arguments that wraps above custom methods in order to  decompose the signal accordingly.

        Returns:
            tuple: breakpoints, labels.
        """
        # get optimal clustering
        n_clusters, clusterer = self.get_optimal_clustering()
        return self.get_breakpoints_and_labels(clusterer, merge=False)



