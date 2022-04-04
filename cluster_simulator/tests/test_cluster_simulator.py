import unittest
import time
import numpy as np
import simpy

from cluster_simulator.application import Cluster, Application, Tier
from cluster_simulator.application import Read_IO_Phase


class TestCluster(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.store = simpy.Store(self.env, capacity=1000)
        nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                           'write': {'seq': 515, 'rand': 505}}
        ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                         'write': {'seq': 100, 'rand': 100}}

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

    def test_tier_init(self):
        print(self.ssd_tier)
        print(self.nvram_tier)

    def test_cluster_init(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        print(cluster)
        self.assertIsInstance(cluster.compute_nodes, simpy.Resource)
        self.assertIsInstance(cluster.compute_cores, simpy.Resource)
        #self.assertIsInstance(cluster.storage_capacity, simpy.Container)
        #self.assertIsInstance(cluster.storage_speed, simpy.Container)

    def test_read_io_phase(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io = Read_IO_Phase(volume=9e9, pattern=0.2)
        # print(read_io)
        self.env.process(read_io.schedule(self.env, cluster, 1))
        self.env.run()

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
