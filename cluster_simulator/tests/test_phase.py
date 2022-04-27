import unittest
import time
import numpy as np
import simpy
from loguru import logger

from cluster_simulator.cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase


class TestPhase(unittest.TestCase):
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
        compute_phase = ComputePhase(duration=10)
        self.env.process(compute_phase.run(self.env, cluster))
        self.env.run()

    def test_read_io_phase(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        #read_io = Read_IOPhase(volume=9e9, pattern=0.2)
        read_io = IOPhase(operation='read', volume=9e9, pattern=0.2)
        self.env.process(read_io.run(self.env, cluster, placement=1))
        self.env.run(until=10)

    def test_write_io_phase(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        #read_io = Read_IOPhase(volume=9e9, pattern=0.2)
        write_io = IOPhase(operation='write', volume=9e9, pattern=0.2)
        self.env.process(write_io.run(self.env, cluster, placement=1))
        self.env.run(until=10)


class TestBandwidthShare(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
                           'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
                         'write': {'seq': 100, 'rand': 100}}

        # logger.remove()

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

    def test_2_read_phases(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        #read_io = Read_IOPhase(volume=9e9, pattern=0.2)
        read_ios = [IOPhase(operation='read', volume=1e9) for i in range(2)]
        for io in read_ios:
            self.env.process(io.run(self.env, cluster, placement=1))  # nvram 200-100

        self.env.run()
        for io in read_ios:
            print(f"app: {io} | bandwidth usage: {io.bandwidth_usage}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
