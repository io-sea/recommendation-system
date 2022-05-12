import unittest
import time
import numpy as np
import simpy
from loguru import logger

from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from analytics import display_run


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

    def test_update_tier(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        # read_io = Read_IOPhase(volume=9e9, pattern=0.2)
        read_io = IOPhase(operation='read', volume=9e9)
        self.env.process(read_io.run(self.env, cluster, placement=1))
        self.env.run(until=10)

    def test_compute_phase(self):
        cluster = Cluster(self.env, compute_nodes=3, cores_per_node=4)
        compute_phase = ComputePhase(duration=10)
        self.env.process(compute_phase.run(self.env, cluster))
        self.env.run()

    def test_read_io_phase(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        # read_io = Read_IOPhase(volume=9e9, pattern=0.2)
        read_io = IOPhase(operation='read', volume=9e9, pattern=0.2)
        self.env.process(read_io.run(self.env, cluster, placement=1))
        self.env.run(until=10)

    def test_write_io_phase(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        # read_io = Read_IOPhase(volume=9e9, pattern=0.2)
        write_io = IOPhase(operation='write', volume=9e9, pattern=0.2)
        self.env.process(write_io.run(self.env, cluster, placement=1))
        self.env.run(until=10)


class TestDataMovement(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                           'write': {'seq': 515, 'rand': 505}}
        ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                         'write': {'seq': 100, 'rand': 100}}

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)


class TestPhaseEphemeralTier(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.data = simpy.Store(self.env)
        self.nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
                                'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
                         'write': {'seq': 100, 'rand': 100}}
        hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                         'write': {'seq': 40, 'rand': 40}}
        self.hdd_tier = Tier(self.env, 'HDD', bandwidth=hdd_bandwidth, capacity=1e12)
        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=self.nvram_bandwidth, capacity=10e9)

    def test_phase_use_bb(self):
        """Test running simple write phase on ephemeral tier."""
        # define an IO phase
        write_io = IOPhase(operation='write', volume=1e9, data=self.data)
        # define burst buffer with its backend PFS
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        # run the phase on the tier with placement = bb
        self.env.process(write_io.run(self.env, cluster, placement=0, use_bb=True))  # nvram 200-100
        self.env.run()
        # ensure at last item that persistent/eph levels are correct
        item = self.data.items[-1]
        self.assertAlmostEqual((item["tier_level"]["HDD"]), 1e9)
        self.assertAlmostEqual((item["BB_level"]), 1e9)

    def test_phase_use_bb_false(self):
        """Test running simple write phase on ephemeral tier."""
        # define an IO phase
        write_io = IOPhase(operation='write', volume=1e9, data=self.data)
        # define burst buffer with its backend PFS
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        # run the phase on the tier with placement = bb
        self.env.process(write_io.run(self.env, cluster, placement=0, use_bb=False))  # nvram 200-100
        self.env.run()
        # ensure at last item that persistent/eph levels are correct
        item = self.data.items[-1]
        self.assertAlmostEqual((item["tier_level"]["HDD"]), 1e9)
        self.assertAlmostEqual((item["BB_level"]), 0)

    def test_phase_use_bb_concurrency(self):
        """Test running simple write phase on ephemeral tier."""
        # define an IO phase
        write_io = IOPhase(operation='write', volume=1e9, data=self.data, appname="Buffered")
        write_io_2 = IOPhase(operation='write', volume=1e9, data=self.data, appname="Concurrent")
        # define burst buffer with its backend PFS
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        # run the phase on the tier with placement = bb
        self.env.process(write_io.run(self.env, cluster, placement=0, use_bb=True))  # nvram 200-100
        self.env.process(write_io_2.run(self.env, cluster, placement=0, use_bb=False))  # nvram 200-100
        self.env.run()
        # ensure at last item that persistent/eph levels are correct
        item = self.data.items[-1]
        self.assertAlmostEqual((item["tier_level"]["HDD"]), 2e9)
        self.assertAlmostEqual((item["BB_level"]), 1e9)

    def test_phase_use_bb_contention(self):
        """Test running simple write phase on ephemeral tier."""
        # define an IO phase
        write_io = IOPhase(operation='write', volume=10e9, data=self.data, appname="Buffered")
        write_io_c = IOPhase(operation='write', volume=10e9, data=self.data, appname="Concurrent")

        # define burst buffer with its backend PFS
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        # run the phase on the tier with placement = bb
        self.env.process(write_io.run(self.env, cluster, placement=0, use_bb=True))  # nvram 400
        self.env.process(write_io_c.run(self.env, cluster, placement=0, use_bb=False))  # hdd 40
        self.env.run()
        # fig = display_run(self.data, cluster, width=800, height=900)
        # fig.show()
        # finally hdd should get 2*10e9
        self.assertAlmostEqual((self.data.items[-1]["tier_level"]["HDD"]), 20e9)
        # ensure at last item that persistent/eph levels are correct
        # self.data.items[-1]
        #self.assertAlmostEqual((item["tier_level"]["HDD"]), 2e9)
        #self.assertAlmostEqual((item["BB_level"]), 1e9)


class TestBandwidthShare(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.data = simpy.Store(self.env)
        self.used_bandwidth = dict()
        nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
                           'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
                         'write': {'seq': 100, 'rand': 100}}

        # logger.remove()

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

    def test_2_read_phases(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        # read_io = Read_IOPhase(volume=9e9, pattern=0.2)
        read_ios = [IOPhase(operation='read', volume=1e9, data=self.data) for i in range(2)]
        placement = 1
        for io in read_ios:
            self.env.process(io.run(self.env, cluster, placement=placement))  # nvram 200-100
        self.env.run()
        tier = get_tier(cluster, placement)
        self.assertAlmostEqual(tier.max_bandwidth["read"]["seq"]/2,
                               self.data.items[0]["bandwidth"])

    def test_2_read_phases_2_placements(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        # read_io = Read_IOPhase(volume=9e9, pattern=0.2)
        read_io_1 = IOPhase(operation='read', volume=1e9, data=self.data)
        read_io_2 = IOPhase(operation='read', volume=1e9, data=self.data)
        self.env.process(read_io_1.run(self.env, cluster, placement="SSD"))  # 200
        self.env.process(read_io_2.run(self.env, cluster, placement="NVRAM"))  # 800
        self.env.run()
        self.assertEqual(set([item["bandwidth"] for item in self.data.items]),
                         {200, 800})

    def test_many_concurrent_read_phases(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        number_of_apps = 3
        read_ios = [IOPhase(operation='read', volume=1e9, data=self.data) for i in range(number_of_apps)]
        for i, io in enumerate(read_ios):
            self.env.process(io.run(self.env, cluster, placement=1, delay=i*0))  # nvram 800

        self.env.run()
        conc = []
        for item in self.data.items:
            conc.append(item["bandwidth_concurrency"])
        self.assertEqual(max(conc), number_of_apps)

    def test_many_concurrent_phases_with_delay(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        number_of_apps = 2
        read_ios = [IOPhase(appname="#"+str(i), operation='read', volume=1e9, data=self.data) for i in range(number_of_apps)]
        for i, io in enumerate(read_ios):
            self.env.process(io.run(self.env, cluster, placement=1, delay=i))  # nvram 800

        self.env.run()
        conc = []
        for item in self.data.items:
            conc.append(item["bandwidth_concurrency"])
        self.assertEqual(max(conc), number_of_apps)

    def test_many_concurrent_phases(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io = IOPhase(appname="#1", operation='read', volume=1e9, data=self.data)
        write_io = IOPhase(appname="#2", operation='write', volume=1e9, data=self.data)
        self.env.process(read_io.run(self.env, cluster, placement=1, delay=0))  # nvram 800
        self.env.process(write_io.run(self.env, cluster, placement=1, delay=0.5))  # nvram 400

        self.env.run()
        conc = []
        for item in self.data.items:
            conc.append(item["bandwidth_concurrency"])
        self.assertEqual(max(conc), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
