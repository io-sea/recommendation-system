import unittest
import time
import numpy as np
import simpy
from loguru import logger

from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model
from cluster_simulator.utils import convert_size, get_tier
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.analytics import display_run


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

    def test_update_tier_write(self):
        """Test that updating tier levels works well when operation is write."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        io = IOPhase(operation='write', volume=10e9, data=data)
        io.env = self.env
        io.update_tier(self.nvram_tier, 5e9)
        self.assertTrue(self.nvram_tier.capacity.level == 5e9)

    def test_update_tier_read(self):
        """Test that updating tier levels works well when operation is read."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        io = IOPhase(operation='read', volume=10e9, data=data)
        io.env = self.env
        io.update_tier(self.nvram_tier, 5e9)
        self.assertTrue(self.nvram_tier.capacity.level == 10e9)

    def test_process_volume_write(self):
        """Test the process volume method."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io = IOPhase(operation='write', volume=9e9, data=data)
        read_io.env = self.env
        ret = self.env.process(read_io.process_volume(step_duration=1,
                                                      volume=1e9,
                                                      available_bandwidth=500,
                                                      cluster=cluster, tier=self.ssd_tier))
        self.env.run()
        self.assertEqual(data.items[0]["volume"], 500)
        self.assertEqual(self.ssd_tier.capacity.level, 500)
        self.assertEqual(ret.value, 1e9 - 500)

    def test_process_volume_read(self):
        """Test the process volume method."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io = IOPhase(operation='read', volume=9e9, data=data)
        read_io.env = self.env
        ret = self.env.process(read_io.process_volume(step_duration=1,
                                                      volume=1e9,
                                                      available_bandwidth=500,
                                                      cluster=cluster, tier=self.ssd_tier))
        self.env.run()
        self.assertEqual(data.items[0]["volume"], 500)
        self.assertEqual(self.ssd_tier.capacity.level, 9e9)
        self.assertEqual(ret.value, 1e9 - 500)

    def test_compute_phase(self):
        cluster = Cluster(self.env, compute_nodes=3, cores_per_node=4)
        compute_phase = ComputePhase(duration=10)
        self.env.process(compute_phase.run(self.env, cluster))
        self.env.run()

    def test_read_io_phase(self):
        data = simpy.Store(self.env)
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io = IOPhase(operation='read', volume=9e9, pattern=0.2)
        self.env.process(read_io.run(self.env, cluster, placement=1))
        self.env.run(until=10)

    def test_write_io_phase(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        write_io = IOPhase(operation='write', volume=9e9, pattern=0.2)
        self.env.process(write_io.run(self.env, cluster, placement=1))
        self.env.run(until=10)


class TestBandwidthShare(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.data = simpy.Store(self.env)
        nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
                           'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
                         'write': {'seq': 100, 'rand': 100}}

        # logger.remove()

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

    def test_2_read_phases(self):
        """Test 2 read phases simultaneously on the same tier, and ensure that bandwidth is /2."""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_ios = [IOPhase(operation='read', volume=1e9, data=self.data) for _ in range(2)]
        placement = 1
        for io in read_ios:
            self.env.process(io.run(self.env, cluster, placement=placement))  # nvram 200-100
        self.env.run()
        tier = get_tier(cluster, placement)
        self.assertAlmostEqual(tier.max_bandwidth["read"]["seq"]/2,
                               self.data.items[0]["bandwidth"])

    def test_2_shifted_read_phases(self):
        """Test 2 read phases simultaneously on the same tier, and ensure that bandwidth is /2."""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io_1 = IOPhase(appname="#1", operation='read', volume=3e9, data=self.data)
        read_io_2 = IOPhase(appname="#2", operation='read', volume=2e9, data=self.data)
        placement = 1  # place data in the same tier
        self.env.process(read_io_1.run(self.env, cluster, placement=placement))  # nvram 200-100
        self.env.process(read_io_2.run(self.env, cluster, placement=placement, delay=2))  # shifted
        self.env.run()
        tier = get_tier(cluster, placement)
        concurrency = [item["bandwidth_concurrency"] for item in self.data.items]
        self.assertListEqual(concurrency, [1, 2, 2, 1])
        self.assertAlmostEqual(tier.capacity.level, max(2e9, 3e9))

    def test_3_shifted_read_phases(self):
        """Test 2 read phases simultaneously on the same tier, and ensure that bandwidth is /2."""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io_1 = IOPhase(appname="#1", operation='read', volume=2e9, data=self.data)
        read_io_2 = IOPhase(appname="#2", operation='read', volume=2e9, data=self.data)
        read_io_3 = IOPhase(appname="#3", operation='read', volume=2e9, data=self.data)
        placement = 1  # place data in the same tier
        self.env.process(read_io_1.run(self.env, cluster, placement=placement))  # nvram 200-100
        self.env.process(read_io_2.run(self.env, cluster, placement=placement, delay=1))  # shifted
        self.env.process(read_io_3.run(self.env, cluster, placement=placement, delay=3))  # shifted
        self.env.run()
        tier = get_tier(cluster, placement)
        # fig = display_run(self.data, cluster, width=800, height=900)
        # fig.show()
        concurrency = [item["bandwidth_concurrency"] for item in self.data.items]
        self.assertListEqual(concurrency, [1, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1])
        self.assertAlmostEqual(tier.capacity.level, max(2e9, 2e9))

    def test_2_shifted_write_phases(self):
        """Test 2 read phases simultaneously on the same tier, and ensure that bandwidth is /2."""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        write_io_1 = IOPhase(appname="#1", operation='write', volume=2e9, data=self.data)
        write_io_2 = IOPhase(appname="#2", operation='write', volume=2e9, data=self.data)
        placement = 1  # place data in the same tier
        self.env.process(write_io_1.run(self.env, cluster, placement=placement))  # nvram 200-100
        self.env.process(write_io_2.run(self.env, cluster, placement=placement, delay=1))  # shifted
        self.env.run()
        tier = get_tier(cluster, placement)
        concurrency = [item["bandwidth_concurrency"] for item in self.data.items]
        self.assertListEqual(concurrency, [1, 2, 2, 1])
        self.assertAlmostEqual(tier.capacity.level, 4e9)

    def test_3_shifted_write_phases(self):
        """Test 2 read phases simultaneously on the same tier, and ensure that bandwidth is /2."""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io_1 = IOPhase(appname="#1", operation='write', volume=2e9, data=self.data)
        read_io_2 = IOPhase(appname="#2", operation='write', volume=2e9, data=self.data)
        read_io_3 = IOPhase(appname="#3", operation='write', volume=2e9, data=self.data)
        placement = 1  # place data in the same tier
        self.env.process(read_io_1.run(self.env, cluster, placement=placement))  # nvram 200-100
        self.env.process(read_io_2.run(self.env, cluster, placement=placement, delay=1))  # shifted
        self.env.process(read_io_3.run(self.env, cluster, placement=placement, delay=3))  # shifted
        self.env.run()
        tier = get_tier(cluster, placement)
        concurrency = [item["bandwidth_concurrency"] for item in self.data.items]
        self.assertListEqual(concurrency, [1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1])
        self.assertAlmostEqual(tier.capacity.level, 6e9, places=5)

    def test_3_shifted_write_phases_diff_placement(self):
        """Test 2 read phases simultaneously on the same tier, and ensure that bandwidth is /2."""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io_1 = IOPhase(appname="#1", operation='write', volume=2e9, data=self.data)
        read_io_2 = IOPhase(appname="#2", operation='write', volume=2e9, data=self.data)
        read_io_3 = IOPhase(appname="#3", operation='write', volume=2e9, data=self.data)

        self.env.process(read_io_1.run(self.env, cluster, placement=1))  # nvram 200-100
        self.env.process(read_io_2.run(self.env, cluster, placement=1, delay=1))  # shifted
        self.env.process(read_io_3.run(self.env, cluster, placement=0, delay=3))  # shifted
        self.env.run()
        tier1 = get_tier(cluster, 1)
        tier0 = get_tier(cluster, 0)
        concurrency = [item["bandwidth_concurrency"] for item in self.data.items]
        self.assertListEqual(concurrency, [1, 2, 2, 2, 2, 1, 1, 1, 1])
        self.assertAlmostEqual(tier1.capacity.level, 4e9, places=5)
        self.assertAlmostEqual(tier0.capacity.level, 2e9, places=5)

    def test_2_decimal_shifted_read_phases(self):
        """Test 2 read phases simultaneously on the same tier, and ensure that bandwidth is /2."""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io_1 = IOPhase(appname="#1", operation='read', volume=2e9, data=self.data)
        read_io_2 = IOPhase(appname="#2", operation='read', volume=2e9, data=self.data)
        placement = 1  # place data in the same tier
        self.env.process(read_io_1.run(self.env, cluster, placement=placement))  # nvram 200-100
        self.env.process(read_io_2.run(self.env, cluster, placement=placement, delay=1.2))  # shifted
        self.env.run()
        tier = get_tier(cluster, placement)
        concurrency = [item["bandwidth_concurrency"] for item in self.data.items]
        self.assertListEqual(concurrency, [1, 2, 2, 2, 2, 1])
        self.assertAlmostEqual(tier.capacity.level, 2e9, places=5)

    def test_2_read_phases_2_placements(self):
        """Test 2 read phases simultaneously on different tiers, and ensure that bandwidth is not /2."""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io_1 = IOPhase(operation='read', volume=1e9, data=self.data)
        read_io_2 = IOPhase(operation='read', volume=1e9, data=self.data)
        self.env.process(read_io_1.run(self.env, cluster, placement="SSD"))  # 200
        self.env.process(read_io_2.run(self.env, cluster, placement="NVRAM"))  # 800
        self.env.run()
        self.assertEqual({item["bandwidth"] for item in self.data.items}, {200, 800})

    def test_many_concurrent_read_phases(self):
        """Test the max number of bandiwdth concurrency is equal to the number of completely overlapping apps"""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        number_of_apps = 3
        read_ios = [IOPhase(operation='read', volume=1e9, data=self.data) for _ in range(number_of_apps)]
        for i, io in enumerate(read_ios):
            self.env.process(io.run(self.env, cluster, placement=1, delay=i*0))  # nvram 800
        self.env.run()
        conc = [item["bandwidth_concurrency"] for item in self.data.items]
        self.assertEqual(max(conc), number_of_apps)

    def test_many_concurrent_phases_with_delay(self):
        """Test the max number of bandiwdth concurrency is equal to the number of partially overlapping apps"""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        number_of_apps = 2
        read_ios = [IOPhase(appname=f"#{str(i)}", operation='read', volume=1e9, data=self.data) for i in range(number_of_apps)]
        for i, io in enumerate(read_ios):
            self.env.process(io.run(self.env, cluster, placement=1, delay=i))  # nvram 800

        self.env.run()
        conc = [item["bandwidth_concurrency"] for item in self.data.items]
        self.assertEqual(max(conc), number_of_apps)

    def test_mix_read_write_concurrent_phases(self):
        """Tests concurrency with mix of read an write phases"""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        read_io = IOPhase(appname="#1", operation='read', volume=1e9, data=self.data)
        write_io = IOPhase(appname="#2", operation='write', volume=1e9, data=self.data)
        self.env.process(read_io.run(self.env, cluster, placement=1, delay=0))  # nvram 800
        self.env.process(write_io.run(self.env, cluster, placement=1, delay=0.5))  # nvram 400

        self.env.run()
        conc = [item["bandwidth_concurrency"] for item in self.data.items]
        self.assertEqual(max(conc), 2)
        self.assertListEqual(conc, [1, 2, 2, 1])


class TestDataMovement(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.data = simpy.Store(self.env)
        hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                         'write': {'seq': 40, 'rand': 40}}
        ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                         'write': {'seq': 100, 'rand': 100}}
        self.hdd_tier = Tier(self.env, 'HDD', bandwidth=hdd_bandwidth, capacity=1e12)
        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier])

    def test_update_tier_on_move(self):
        """Test if updating tiers levels after data moves works well."""
        # prepare initial volume on tier
        self.hdd_tier.capacity.put(20e9)
        self.assertEqual(self.ssd_tier.capacity.level, 0)
        write_io = IOPhase(operation='write', volume=10e9, data=self.data)
        write_io.update_tier_on_move(source_tier=self.hdd_tier,
                                     target_tier=self.ssd_tier,
                                     volume=10e9,
                                     erase=False)
        self.assertEqual(self.ssd_tier.capacity.level, 10e9)

    def test_move_volume_write(self):
        """Test moving volume on I/O of type write."""
        write_io = IOPhase(operation='write', volume=9e9, data=self.data)
        write_io.env = self.env
        self.hdd_tier.capacity.put(20e9)

        ret = self.env.process(write_io.move_volume(step_duration=1,
                                                    volume=1e9,
                                                    available_bandwidth=500e6,
                                                    cluster=self.cluster,
                                                    source_tier=self.hdd_tier,
                                                    target_tier=self.ssd_tier))
        self.env.run()
        self.assertEqual(self.data.items[0]["volume"], 500e6)
        self.assertEqual(self.ssd_tier.capacity.level, 500e6)

    def test_move_volume_write_with_erase(self):
        """Test moving volume on I/O of type write with erase option."""
        write_io = IOPhase(operation='write', volume=9e9, data=self.data)
        write_io.env = self.env
        self.hdd_tier.capacity.put(20e9)
        self.assertEqual(self.hdd_tier.capacity.level, 20e9)
        ret = self.env.process(write_io.move_volume(step_duration=1,
                                                    volume=1e9,
                                                    available_bandwidth=500e6,
                                                    cluster=self.cluster,
                                                    source_tier=self.hdd_tier,
                                                    target_tier=self.ssd_tier,
                                                    erase=True))
        self.env.run()
        self.assertEqual(self.data.items[0]["volume"], 500e6)
        self.assertEqual(self.ssd_tier.capacity.level, 500e6)
        self.assertEqual(self.hdd_tier.capacity.level, 20e9-500e6)
        # self.assertEqual(ret.value, 1e9 - 500)

    def test_update_tier_on_move_with_erase(self):
        """Test if updating tiers levels after data moves works well with erase option."""
        # prepare initial volume on tier
        self.hdd_tier.capacity.put(20e9)
        self.assertEqual(self.hdd_tier.capacity.level, 20e9)
        self.assertEqual(self.ssd_tier.capacity.level, 0)
        write_io = IOPhase(operation='write', volume=10e9, data=self.data)
        write_io.update_tier_on_move(source_tier=self.hdd_tier,
                                     target_tier=self.ssd_tier,
                                     volume=10e9,
                                     erase=True)
        self.assertEqual(self.ssd_tier.capacity.level, 10e9)
        self.assertEqual(self.hdd_tier.capacity.level, 20e9-10e9)

    def test_move_step(self):
        """Test that move step moves data from tier to another for the IO volume."""
        self.hdd_tier.capacity.put(20e9)
        self.assertEqual(self.hdd_tier.capacity.level, 20e9)
        self.assertEqual(self.ssd_tier.capacity.level, 0)

        write_io = IOPhase(operation='write', volume=10e9, data=self.data)
        ret = self.env.process(write_io.move_step(self.env, self.cluster, self.hdd_tier, self.ssd_tier))
        self.env.run()
        self.assertEqual(self.hdd_tier.capacity.level, 10e9)
        self.assertEqual(self.ssd_tier.capacity.level, 10e9)
        self.assertEquals(self.data.items[0]["type"], "movement")
        self.assertEquals(self.data.items[0]["data_placement"]["source"], self.hdd_tier.name)

    def test_concurrent_move_step(self):
        """Test that move step moves data from tier to another for the IO volume and adapts bandwidth in case of concurrency."""
        self.hdd_tier.capacity.put(20e9)
        self.assertEqual(self.hdd_tier.capacity.level, 20e9)
        self.assertEqual(self.ssd_tier.capacity.level, 0)

        io1 = IOPhase(operation='write', volume=10e9, data=self.data)
        io2 = IOPhase(operation='write', volume=10e9, data=self.data)
        self.env.process(io1.move_step(self.env, self.cluster, self.hdd_tier, self.ssd_tier))
        self.env.process(io2.move_step(self.env, self.cluster, self.hdd_tier, self.ssd_tier))
        self.env.run()
        self.assertEqual(self.hdd_tier.capacity.level, 0)
        self.assertEqual(self.ssd_tier.capacity.level, 20e9)

    def test_io_concurrent_move_step(self):
        """Test that move step moves data from tier to another for the IO volume and adapts bandwidth in case of concurrency. Here the concurrent IO is a write on the target tier."""
        self.hdd_tier.capacity.put(20e9)
        self.assertEqual(self.hdd_tier.capacity.level, 20e9)
        self.assertEqual(self.ssd_tier.capacity.level, 0)

        io1 = IOPhase(operation='write', volume=10e9, data=self.data)
        io2 = IOPhase(operation='write', volume=10e9, data=self.data)
        self.env.process(io1.move_step(self.env, self.cluster, self.hdd_tier, self.ssd_tier))
        # write a concurrent IO on ssd_tier
        self.env.process(io2.run(self.env, self.cluster, placement=1))
        self.env.run()
        self.assertEqual(self.hdd_tier.capacity.level, 10e9)
        self.assertEqual(self.ssd_tier.capacity.level, 20e9)

    def test_shifted_io_concurrent_move_step(self):
        """Test that move step moves data from tier to another for the IO volume and adapts bandwidth in case of concurrency. Here the concurrent IO is time_shifted write on the target tier."""
        self.hdd_tier.capacity.put(20e9)
        self.assertEqual(self.hdd_tier.capacity.level, 20e9)
        self.assertEqual(self.ssd_tier.capacity.level, 0)

        io1 = IOPhase(operation='write', volume=10e9, data=self.data)
        io2 = IOPhase(operation='write', volume=10e9, data=self.data)
        self.env.process(io1.move_step(self.env, self.cluster, self.hdd_tier, self.ssd_tier))
        # write a concurrent IO on ssd_tier
        self.env.process(io2.run(self.env, self.cluster, placement=1, delay=10))
        self.env.run()
        self.assertEqual(self.hdd_tier.capacity.level, 10e9)
        self.assertEqual(self.ssd_tier.capacity.level, 20e9)


class TestPhaseEphemeralTier(unittest.TestCase):
    """Test phases that happens on tier of type Ephemeral."""

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

    def test_process_volume_use_bb(self):
        """Test running process volume phase on ephemeral tier and checks buffer level."""
        data = simpy.Store(self.env)
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        write_io = IOPhase(operation='write', volume=9e9, data=data)
        write_io.env = self.env
        ret = self.env.process(write_io.process_volume(step_duration=1,
                                                       volume=1e9,
                                                       available_bandwidth=500e6,
                                                       cluster=cluster, tier=bb))
        self.env.run()
        self.assertEqual(data.items[0]["volume"], 500e6)
        self.assertEqual(bb.capacity.level, 500e6)
        self.assertEqual(ret.value, 1e9 - 500e6)

    def test_move_volume_use_bb(self):
        """Test running move volume phase on ephemeral tier and checks buffer level."""
        data = simpy.Store(self.env)
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        write_io = IOPhase(operation='write', volume=9e9, data=data)
        write_io.env = self.env
        ret = self.env.process(write_io.move_volume(step_duration=1,
                                                    volume=1e9,
                                                    available_bandwidth=500e6,
                                                    cluster=cluster,
                                                    source_tier=self.hdd_tier,
                                                    target_tier=self.ssd_tier))
        self.env.run()
        self.assertEqual(data.items[0]["volume"], 500e6)
        self.assertEqual(bb.capacity.level, 500e6)
        self.assertEqual(ret.value, 1e9 - 500e6)

    def test_phase_use_bb(self):
        """Test running complete write phase on ephemeral tier and checks destaging buffered data on persistent tier."""
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
        self.assertEqual(self.data.items[0]["volume"], 1e9)
        self.assertEqual(self.data.items[-1]["volume"], 1e9)
        self.assertEqual(bb.capacity.level, 1e9)
        self.assertEqual(bb.persistent_tier.capacity.level, 1e9)

    def test_phase_use_bb_false(self):
        """Test running simple write phase on ephemeral tier with False option and checks if it runs without buffering."""
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
        """Test two writing phases, one has burst buffer usage and the other not. The two phases are happening concurrently."""
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
        # fig = display_run(self.data, cluster, width=800, height=900)
        # fig.show()

    def test_phase_use_bb_contention(self):
        """Test two writing phases, one has burst buffer usage and the other not. The two phases are happening concurrently. The one writing in BB will overlfow data"""
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
        # self.assertAlmostEqual((self.data.items[-1]["tier_level"]["HDD"]), 20e9)
        # ensure at last item that persistent/eph levels are correct
        # self.data.items[-1]
        #self.assertAlmostEqual((item["tier_level"]["HDD"]), 2e9)
        #self.assertAlmostEqual((item["BB_level"]), 1e9)
        # fig = display_run(self.data, cluster, width=800, height=900)
        # fig.show()


if __name__ == '__main__':
    unittest.main(verbosity=2)
