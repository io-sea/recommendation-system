import unittest
import time
import numpy as np
import simpy

from cluster_simulator.application import Cluster, Application


class TestCluster(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.store = simpy.Store(self.env, capacity=1000)

    def test_cluster_init(self):
        cluster = Cluster(self.env)
        self.assertIsInstance(cluster.compute_cores, simpy.Resource)
        self.assertIsInstance(cluster.storage_capacity, simpy.Container)
        self.assertIsInstance(cluster.storage_speed, simpy.Container)

    def test_application_basic(self):
        # Simple app: read 1GB -> compute 10s -> write 5GB
        compute = [0, 10]
        read = [1e9, 0]
        write = [0, 5e9]
        app = Application(self.env, self.store,
                          compute=compute,
                          read=read,
                          write=write)
        print(app.store.capacity)
        print(app.store.items)
        self.assertEqual(len(app.store.items), 3)
