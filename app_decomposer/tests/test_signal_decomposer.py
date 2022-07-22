import unittest
import time
import numpy as np
import simpy
from loguru import logger

from app_decomposer.signal_decomposer import KmeansSignalDecomposer

class TestPhase(unittest.TestCase):    

    def test_kmeans_decomposer_init(self):
        signal = np.array([0, 1, 2, 3])
        kmeans = KmeansSignalDecomposer(signal)
        np.testing.assert_array_equal(kmeans.signal, signal)
        
    def test_kmeans_decomposer_init(self):
        signal = np.array([0, 1, 2, 3])
        ksd = KmeansSignalDecomposer(signal)
        n_clusters = ksd.get_optimal_n_clusters()
        print(n_clusters)