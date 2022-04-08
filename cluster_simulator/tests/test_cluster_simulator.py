import unittest
import time
import numpy as np
import simpy

from cluster_simulator.cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.application import Application, IO_Phase, IO_Compute


class TestCluster(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
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

    def test_compute_phase(self):
        cluster = Cluster(self.env, compute_nodes=3, cores_per_node=4)
        compute_phase = IO_Compute(duration=10)
        self.env.process(compute_phase.run(self.env, cluster))
        self.env.run()

    def test_read_io_phase(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        #read_io = Read_IO_Phase(volume=9e9, pattern=0.2)
        read_io = IO_Phase(operation='read', volume=9e9, pattern=0.2)
        self.env.process(read_io.run(self.env, cluster, placement=1))
        self.env.run()

    def test_write_io_phase(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        #read_io = Read_IO_Phase(volume=9e9, pattern=0.2)
        write_io = IO_Phase(operation='write', volume=9e9, pattern=0.2)
        self.env.process(write_io.run(self.env, cluster, placement=1))
        self.env.run()

    def test_application_init(self):
        # Simple app: read 1GB -> compute 10s -> write 5GB
        compute = [0, 10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [0, 1]
        app = Application(self.env,
                          compute=compute,
                          read=read,
                          write=write)
        # print(app.store.capacity)
        # print(app.store.items)
        self.assertEqual(len(app.store.items), 3)

    def test_application_run(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        # Simple app: read 1GB -> compute 10s -> write 5GB
        compute = [0, 10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [0, 1]
        app = Application(self.env,
                          compute=compute,
                          read=read,
                          write=write)

        self.env.process(app.run(cluster, tiers=tiers))
        self.env.run()
        # self.env.run(until=25)
        #self.assertEqual(len(app.store.items), 3)


class TestApps(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                           'write': {'seq': 515, 'rand': 505}}
        ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                         'write': {'seq': 100, 'rand': 100}}

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

    def test_app_simple(self):
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0, 10],
                           read=[1e9, 0], write=[0, 5e9], data=data)
        self.env.process(app1.run(cluster, tiers=[0, 0]))
        self.env.run()
        for item in data.items:
            print(item)

    def test_app_pure_read(self):
        data = simpy.Store(self.env)
        read_size = 1e9
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0],
                           read=[read_size], write=[0], data=data)
        self.env.process(app1.run(cluster, tiers=[0]))
        self.env.run()
        self.assertEqual(len(data.items), 1)
        tier_name = data.items[0]["data_placement"]["placement"]
        self.assertEqual(data.items[0]["tier_level"][tier_name], read_size)

    def test_app_pure_write(self):
        data = simpy.Store(self.env)
        write_size = 1e9
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0],
                           read=[0], write=[write_size], data=data)
        self.env.process(app1.run(cluster, tiers=[0]))
        self.env.run()
        self.assertEqual(len(data.items), 1)
        tier_name = data.items[0]["data_placement"]["placement"]
        self.assertEqual(data.items[0]["tier_level"][tier_name], write_size)


class TestAppsSuperposition(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                           'write': {'seq': 515, 'rand': 505}}
        ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                         'write': {'seq': 100, 'rand': 100}}
        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

    def test_2_computes_sequential(self):
        """Two compute phases that should run sequentially because the cluster
        has only one core."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=1,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0, 10],
                           read=[0, 0], write=[0, 0], data=data)
        app2 = Application(self.env, compute=[0, 15],
                           read=[0, 0], write=[0, 0], data=data)
        self.env.process(app1.run(cluster, tiers=[0, 0]))
        self.env.process(app2.run(cluster, tiers=[0, 0]))
        self.env.run()
        timespan = 0.0
        for item in data.items:
            timespan = max(timespan, item["t_end"])
        self.assertEqual(timespan, 25.0)

    def test_2_computes_parallel(self):
        """Two compute phases that should run in parallel because the cluster
        has now 2 cores."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0, 10],
                           read=[0, 0], write=[0, 0], data=data)
        app2 = Application(self.env, compute=[0, 15],
                           read=[0, 0], write=[0, 0], data=data)
        self.env.process(app1.run(cluster, tiers=[0, 0]))
        self.env.process(app2.run(cluster, tiers=[0, 0]))
        self.env.run()
        timespan = 0.0
        for item in data.items:
            timespan = max(timespan, item["t_end"])
        self.assertEqual(timespan, 15.0)

    def test_2_IO_sequential(self):
        """Two I/O phases that should run sequentially because the cluster
        has only one core."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=1,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0],
                           read=[1e9], write=[0], data=data)
        app2 = Application(self.env, compute=[0],
                           read=[2e9], write=[0], data=data)
        self.env.process(app1.run(cluster, tiers=[0, 0]))
        self.env.process(app2.run(cluster, tiers=[0, 0]))
        self.env.run()
        self.assertEqual(data.items[0]["t_end"], data.items[1]["t_start"])

    def test_2_IO_parallel(self):
        """Two I/O phases that should run in parallel because the cluster
        has2 cores."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0],
                           read=[1e9], write=[0], data=data)
        app2 = Application(self.env, compute=[0],
                           read=[2e9], write=[0], data=data)
        self.env.process(app1.run(cluster, tiers=[0, 0]))
        self.env.process(app2.run(cluster, tiers=[0, 0]))
        self.env.run()
        self.assertEqual(data.items[0]["t_start"], data.items[1]["t_start"])
