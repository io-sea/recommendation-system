import unittest
import time
import numpy as np
from loguru import logger
from sklearn.cluster import KMeans

from app_decomposer.signal_decomposer import KmeansSignalDecomposer, get_lowest_cluster, arrange_labels

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

    def test_get_lowest_cluster_1(self):
        """Test if get lowest cluster method works well."""
        labels = np.array([1, 1, 7, 7, 1])
        signal = np.array([3, 2, 1, 1, 4])
        label0 = get_lowest_cluster(labels, signal)
        self.assertEqual(label0, 7)

    def test_get_lowest_cluster_2(self):
        """Test if get lowest cluster method works well."""
        labels = np.array([0, 0, 1, 1, 0]) # two labels
        signal = np.array([1, 2, 3, 4, 5]) # label0 avg~2, label1 avg~3.5
        label0 = get_lowest_cluster(labels, signal)
        self.assertEqual(label0, 0)

    def test_arrange_labels_0(self):
        """Test that labels arrangement works properly."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        label0 = 1
        arranged_labels =arrange_labels(labels, label0)
        # step1: [3, 3, 1, 1, 2, 2]
        # step2: [3, 3, 0, 0, 2, 2]
        # step3: [1, 1, 0, 0, 2, 2]
        self.assertListEqual(arranged_labels.tolist(), [1, 1, 0, 0, 2, 2])

    def test_arrange_labels_1(self):
        """Test that labels arrangement works properly when labels0 = 0."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        label0 = 0
        arranged_labels =arrange_labels(labels, label0)
        # step1: [3, 3, 1, 1, 2, 2]
        # step2: [3, 3, 0, 0, 2, 2]
        # step3: [1, 1, 0, 0, 2, 2]
        print(arranged_labels)
        self.assertListEqual(arranged_labels.tolist(), [0, 0, 1, 1, 2, 2])

    def test_arrange_labels_with_get_lowest(self):
        """Test that labels arranges the lowest to 0."""
        signal = np.array([10, 12, 4, 5, 20, 25])
        labels = np.array([1, 1, 2, 2, 0, 0])
        label0 = get_lowest_cluster(labels, signal)
        arranged_labels =arrange_labels(labels, label0)
        # step1: [1, 1, 2, 2, 3, 3]
        # step2: [1, 1, 0, 0, 3, 3]
        # step3: [1, 1, 0, 0, 2, 2]
        self.assertEqual(label0, 2)
        self.assertListEqual(arranged_labels.tolist(), [1, 1, 0, 0, 2, 2])

    def test_arrange_labels_with_get_lowest_no_zero(self):
        """Test that labels arranges the lowest to 0."""
        signal = np.array([10, 12, 4, 5, 20, 25])
        labels = np.array([1, 1, 4, 4, 2, 2])
        label0 = get_lowest_cluster(labels, signal)
        arranged_labels =arrange_labels(labels, label0)
        # step1: [1, 1, 2, 2, 3, 3]
        # step2: [1, 1, 0, 0, 3, 3]
        # step3: [1, 1, 0, 0, 2, 2]
        self.assertEqual(label0, 4)
        self.assertListEqual(arranged_labels.tolist(), [4, 4, 0, 0, 2, 2])


    def test_kmeans_decomposer_signal_dim_lower_n_clusters(self):
        """Tests that kmeans decomposer stops incrementing n_clusters at signal length."""
        signal = np.arange(3).reshape(-1, 1) # shape is (n, 1)
        ksd = KmeansSignalDecomposer(signal)
        n_clusters, _ = ksd.get_optimal_clustering()
        self.assertEqual(3, n_clusters)

    def test_kmeans_decomposer_signal_dim_lower_n_clusters_long_signal(self):
        """Tests that kmeans decomposer stops incrementing n_clusters at signal length."""
        signal = np.arange(500).reshape(-1, 1) # shape is (n, 1)
        ksd = KmeansSignalDecomposer(signal)
        n_clusters, clusterer = ksd.get_optimal_clustering()
        self.assertIsInstance(clusterer, KMeans)
        self.assertEqual(5, n_clusters)

    def test_get_breakpoints_and_labels_merge(self):
        """Tests that returns are lists of breakpoints and labels when merge=True.
        In this case, there is 1 breakpoint and two values for labels."""
        signal = np.arange(500).reshape(-1, 1) # shape is (n, 1)
        ksd = KmeansSignalDecomposer(signal)
        n_clusters, clusterer = ksd.get_optimal_clustering()
        bkps, labels = ksd.get_breakpoints_and_labels(clusterer, merge=True)
        self.assertEqual(len(bkps), 2)
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
        self.assertListEqual(breakpoints, [2, 5, 7, 8])
        self.assertListEqual(labels.tolist(), [1, 1, 3, 3, 3, 2, 2, 4, 0, 0])

    def test_decompose_complexity(self):
        """Functional test that decomposes a signal and returns a list of breakpoints and labels."""
        N = 1000
        signal = np.random.random(size=(N,1)).reshape(-1, 1)
        start_time = time.time()
        decomposer = KmeansSignalDecomposer(signal)
        breakpoints, labels = decomposer.decompose()
        print(f"elapsed_time for N={N} :  {time.time() - start_time}")



if __name__ == '__main__':
    unittest.main(verbosity=2)
