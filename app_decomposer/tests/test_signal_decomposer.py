import unittest
import time
import numpy as np
import simpy
from loguru import logger
import sklearn

from app_decomposer.signal_decomposer import KmeansSignalDecomposer

class TestKmeansSignalDecomposer(unittest.TestCase):
    """Test that the app decomposer follows some pattern."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def test_kmeans_decomposer_init(self):
        """Tests that attribute is registered."""
        signal = np.arange(3).reshape(-1, 1)
        kmeans_decomposer = KmeansSignalDecomposer(signal)
        # test attribute
        np.testing.assert_array_equal(kmeans_decomposer.signal, signal)

    def test_kmeans_decomposer_init_with_list(self):
        """Tests that attribute is registered."""
        signal = list(range(10))
        kd = KmeansSignalDecomposer(signal)
        self.assertEqual(kd.signal.shape, (10, 1))

    def test_kmeans_decomposer_init_with_iterable(self):
        """Tests that attribute is registered."""
        signal = range(10)
        kd = KmeansSignalDecomposer(signal)
        self.assertEqual(kd.signal.shape, (10, 1))


    def test_kmeans_decomposer_signal_dim_lower_n_clusters(self):
        """Tests that kmeans decomposer stops incrementing n_clusters at signal length."""
        signal = np.arange(3).reshape(-1, 1) # shape is (n, 1)
        ksd = KmeansSignalDecomposer(signal)
        n_clusters = ksd.get_optimal_n_clusters()
        self.assertEqual(3, n_clusters)

    def test_kmeans_decomposer_signal_dim_lower_n_clusters(self):
        """Tests that kmeans decomposer stops incrementing n_clusters at signal length."""
        signal = np.arange(500).reshape(-1, 1) # shape is (n, 1)
        ksd = KmeansSignalDecomposer(signal)
        n_clusters, clusterer = ksd.get_optimal_clustering()
        self.assertIsInstance(clusterer, sklearn.cluster._kmeans.KMeans)
        self.assertEqual(12, n_clusters)

    def test_get_breakpoints_and_labels_merge(self):
        """Tests that returns are lists of breakpoints and labels when merge=True.
        In this case, there is 1 breakpoint and two values for labels."""
        signal = np.arange(500).reshape(-1, 1) # shape is (n, 1)
        ksd = KmeansSignalDecomposer(signal)
        n_clusters, clusterer = ksd.get_optimal_clustering()
        bkps, labels = ksd.get_breakpoints_and_labels(clusterer, merge=True)
        self.assertEqual(len(bkps), 1)
        self.assertListEqual(np.unique(labels).tolist(), [0, 1])

    def test_get_breakpoints_and_labels(self):
        """Tests that returns are lists of breakpoints and labels when merge=False.
        In this case, there is 1 breakpoint and two values for labels."""
        signal = np.array([0]*10+[1]*30).reshape(1, -1) # shape is (n, 1)
        ksd = KmeansSignalDecomposer(signal)
        n_clusters, clusterer = ksd.get_optimal_clustering()
        bkps, labels = ksd.get_breakpoints_and_labels(clusterer)
        self.assertEqual(len(bkps), 1)
        self.assertListEqual(np.unique(labels).tolist(), [0, 1])

    def test_decompose(self):
        """Functional test that decomposes a signal and returns a list of breakpoints and labels."""
        signal = np.arange(10).reshape(-1, 1)
        decomposer = KmeansSignalDecomposer(signal)
        breakpoints, labels = decomposer.decompose()
        self.assertListEqual(breakpoints, [2, 3, 5, 6, 8])
        self.assertListEqual(labels.tolist(), [0, 0, 4, 2, 2, 5, 3, 3, 1, 1])






if __name__ == '__main__':
    unittest.main(verbosity=2)
