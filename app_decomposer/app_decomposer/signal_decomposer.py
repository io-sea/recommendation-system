#!/usr/bin/env python
"""
This module proposes a class and methods to slice a 1D timeserie (signal) into a set of parts (phases) separated by breakpoints.
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
import pwlf
import warnings

# TODO: implement and hdbscan version for clustering
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
        label0 (int): label with the lowest values.
    """
    unique_labels = np.unique(labels).tolist()
    mean_cluster_values = [np.mean(signal[np.where(labels==label, True, False)]) for label in unique_labels]

    return unique_labels[mean_cluster_values.index(np.min(mean_cluster_values))]


def arrange_labels(labels, label0):
    """Rearrange labels so that label0 get the 0-label.

    Args:
        labels (ndarray): rearranged labels
        label0 (int): number of the label having le lowest values in signal

    Retuns:
        label0 (int): label with the lowest values.
    """
    min_label = min(np.unique(labels))
    max_label = max(np.unique(labels))
    # do arrangement if not already done
    if label0 != 0:
        # replace min_label by max_label+1
        labels = np.array([max_label+1 if item == min_label else item for item in labels])
        # replace label0 by 0
        labels = np.array([0 if item == label0 else item for item in labels])
        # replace max_label+1 by label0
        labels = np.array([label0 if item == max_label+1 else item for item in labels])

    return labels


class KmeansSignalDecomposer(SignalDecomposer):
    """Implements signal decomposer based on kmeans clustering."""
    def __init__(self, signal, v0_threshold=0.05, merge=False):
        """Initializes a kmeans clustering based decomposer.

        Args:
            signal (iterable): 1d signal to decompose.
            v0_threshold (float, optional): relative mean value of the weight of the lowest level of cluster point ordinates. Defaults to 0.05.
            merge (bool, optional): If True merges all labels > 0 into one label. Defaults to False.
        """
        # convert any iterable into numpy array of size (n, 1)
        self.signal = np.array(list(signal)).reshape(-1, 1)
        self.v0_threshold = v0_threshold
        self.merge = merge


    def get_optimal_clustering(self):
        """Get the optimal number of clusters when cluster0 average values are below v0_threshold comparing to total mean values. The clustering is done in the 1-dimension that carries the dataflow signal.

        Args:
            v0_threshold (float, optional): relative mean value of the average lowest level of cluster point ordinates. Defaults to 0.05.

        Returns:
            tuple (int, sklearn.cluster.KMeans): return the optimal number of cluster complying the threshold and the related KMeans object.
        """
        n_clusters = 1
        v0_weight = 1
        while v0_weight > self.v0_threshold:
            n_clusters += 1
            if n_clusters > len(self.signal):
                # if n_clusters exceeds signal points, exit the loop
                break
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.signal)
            label0 = get_lowest_cluster(kmeans.labels_, self.signal)
            v0_indexes = np.where(kmeans.labels_==label0, True, False)
            v0_weight = self.signal[v0_indexes].sum() / self.signal.sum()


        return n_clusters, kmeans

    def get_breakpoints_and_labels(self, kmeans):
        """Returns breakpoints and labels from clustering algorithm. Labels could be merged
        Args:
            kmeans (sklearn.cluster.KMeans): clustering object
            merge (bool, optional): if True labels > 0 will be merged into one label. Defaults to False.

        Returns:
            (breakpoints, labels): list of breakpoints where each element is the index that separates two phases. Each phase get assigned a label to identify its nature later.
        """
        ab = np.arange(len(self.signal))
        label0 = get_lowest_cluster(kmeans.labels_, self.signal)
        labels =arrange_labels(kmeans.labels_, label0)

        labels = np.where(labels > 0, 1, 0) if self.merge else labels
        return  ab[np.insert(np.where(np.diff(labels)!=0, True, False), 0, False)].tolist(), labels


    def decompose(self):
        """Common method without proper arguments that wraps above custom methods in order to  decompose the signal accordingly.

        Returns:
            tuple: breakpoints, labels.
        """
        # get optimal clustering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_clusters, clusterer = self.get_optimal_clustering()
        return self.get_breakpoints_and_labels(clusterer)

    def reconstruct(self, breakpoints):
        """Method to get a reconstructed piecewise linear signal from breakpoints.

        Args:
            breakpoints (list): list of indices where labels changes.

        Returns:
            _type_: _description_
        """
        ab = np.arange(len(self.signal))
        my_pwlf = pwlf.PiecewiseLinFit(ab, self.signal, degree=0)
        bkps = np.array([min(ab)] + list(map(lambda x: x-1, breakpoints)) + [max(ab)])
        my_pwlf.fit_with_breaks(bkps)
        return my_pwlf.predict(ab)



