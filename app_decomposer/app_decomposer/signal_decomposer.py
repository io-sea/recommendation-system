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

class SignalDecomposer(ABC):
    """Abstract class for signal decomposer module. """
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def decompose(self):
        """Method to return breakpoints."""
        pass
    

class KmeansSignalDecomposer(SignalDecomposer):
    """Implements signal decomposer based on kmeans clustering."""
    def __init__(self, signal):
        self.signal = signal
    
    def get_optimal_n_clusters(self, v0_threshold = 0.05):
        n_clusters = 2
        v0_weight = 1
        while v0_weight > v0_threshold:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.signal)
            v0_indexes = np.where(kmeans.labels_==0, True, False)
            v0_weight = signal[v0_indexes].sum() / signal.sum()
            n_clusters += 1
            if n_clusters >= len(signal):
                # if n_clusters equals signal points, exit the loop
                break
            
        return n_clusters
        
    def decompose(self):
        """Method to return breakpoints."""
        pass
    
        
        