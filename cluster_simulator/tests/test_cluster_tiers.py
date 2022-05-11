import unittest
import time
import numpy as np
import simpy
from loguru import logger

from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase


class TestClusterTiers(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
                                'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
                         'write': {'seq': 100, 'rand': 100}}
        hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                         'write': {'seq': 40, 'rand': 40}}
        self.hdd_tier = Tier(self.env, 'HDD', bandwidth=hdd_bandwidth, capacity=1e12)
        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=self.nvram_bandwidth, capacity=10e9)

    def test_get_tier_as_object_and_recursive(self):
        """Test reaching tier object using object"""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier])
        result = get_tier(cluster, tier_reference=self.hdd_tier)
        result2 = get_tier(cluster, result)
        self.assertIsInstance(result, Tier)
        self.assertEquals(result, self.hdd_tier)
        self.assertEquals(result2, self.hdd_tier)

    def test_get_tier_as_None(self):
        """Test reaching tier object using None as tier_reference generates Exception."""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier])
        with self.assertRaises(Exception):
            result = get_tier(cluster, tier_reference=None)

    def test_get_tier_as_int_reference(self):
        """Test reaching tier object using object"""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier])
        result = get_tier(cluster, tier_reference=1)
        self.assertEquals(result, self.ssd_tier)

    def test_get_tier_name_no_bb(self):
        """Test reaching tier object using integer or string"""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier])
        result = get_tier(cluster, tier_reference="HDD")
        self.assertEquals(result.name, "HDD")
        result = get_tier(cluster, 0)
        self.assertEquals(result.name, "HDD")

    def test_get_tier_name_with_bb(self):
        """Test reaching tier object using integer or string"""
        # define burst buffer with its backend PFS
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        result = get_tier(cluster, tier_reference=0, use_bb=True)
        self.assertEquals(result.name, "BB")
        self.assertIsInstance(result, EphemeralTier)
        # retrieve bb backend tier
        self.assertEqual(result.persistent_tier, self.hdd_tier)

    def test_get_tier_with_bb(self):
        """Test reaching tier object using integer or string"""
        # define burst buffer with its backend PFS
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        result = get_tier(cluster, tier_reference="BB", use_bb=True)
        self.assertEquals(result.name, "BB")
        self.assertIsInstance(result, EphemeralTier)
        # retrieve bb backend tier
        self.assertEqual(result.persistent_tier, self.hdd_tier)

    def test_create_transient_tier(self):
        """Test init and creation of a tier"""
        sbb_tier = EphemeralTier(self.env, name='SBB',
                                 persistent_tier=self.ssd_tier,
                                 bandwidth=self.nvram_bandwidth,
                                 capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=sbb_tier)
        self.assertIsInstance(cluster.ephemeral_tier, EphemeralTier)


class TestClusterDataMovement(unittest.TestCase):
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

        self.SBB = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                                 bandwidth=self.nvram_bandwidth, capacity=10e9)
        self.cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                               ephemeral_tier=self.SBB)

    def test_init(self):
        self.assertIsInstance(self.cluster, Cluster)
        self.assertIsInstance(self.SBB, EphemeralTier)
        for tier in self.cluster.tiers:
            self.assertIsInstance(tier, Tier)

    def test_get_max_bandwidth(self):
        tier = get_tier(self.cluster, "HDD")
        bw = self.cluster.get_max_bandwidth(tier, operation="read")/1e6
        self.assertEqual(bw, 80)
        tier = get_tier(self.cluster, "SSD")
        bw = self.cluster.get_max_bandwidth(tier, operation="read")/1e6
        self.assertEqual(bw, 200)
        tier = 0
        bw = self.cluster.get_max_bandwidth(tier, operation="read")/1e6
        self.assertEqual(bw, 80)

    def test_move_data_simple_no_erase(self):
        """Test that for simple data movement the retained bandwidth is the bottelneck of read and write"""

        move_data_event = self.cluster.move_data(self.env, self.hdd_tier, self.ssd_tier,
                                                 total_volume=3e9, erase=False, data=self.data)
        self.env.process(move_data_event)
        self.env.run()
        last_item = self.data.items[-1]
        self.assertEqual(last_item["bandwidth"], 80)
        self.assertEqual(last_item["tier_level"]["HDD"], 3e9)
        self.assertEqual(last_item["tier_level"]["SSD"], 3e9)

    def test_move_data_simple_with_erase(self):
        """Test that for simple data movement the retained bandwidth is the bottelneck of read and write"""
        move_data_event = self.cluster.move_data(self.env, self.hdd_tier, self.ssd_tier,
                                                 total_volume=3e9,
                                                 erase=True, data=self.data)
        self.env.process(move_data_event)
        self.env.run()
        last_item = self.data.items[-1]
        self.assertEqual(last_item["bandwidth"], 80)
        self.assertEqual(last_item["tier_level"]["HDD"], 0)
        self.assertEqual(last_item["tier_level"]["SSD"], 3e9)

    def test_move_data_another_tier_with_erase(self):
        """Test that for simple data movement the retained bandwidth is the bottelneck of read and write"""

        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier],
                          ephemeral_tier=self.SBB)
        move_data_event = cluster.move_data(self.env, self.ssd_tier, self.nvram_tier,
                                            total_volume=3e9, erase=True, data=self.data)
        self.env.process(move_data_event)
        self.env.run()
        last_item = self.data.items[-1]
        self.assertEqual(last_item["bandwidth"], 200)
        self.assertEqual(last_item["tier_level"]["SSD"], 0)
        self.assertEqual(last_item["tier_level"]["NVRAM"], 3e9)

    def test_move_data_concurrency_1_with_erase(self):
        """Test that for simple data movement the retained bandwidth is the bottelneck of read and write"""
        # self.nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
        #                         'write': {'seq': 400, 'rand': 400}}
        # ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
        #                  'write': {'seq': 100, 'rand': 100}}
        # hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
        #                  'write': {'seq': 40, 'rand': 40}}

        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier, self.nvram_tier],
                          ephemeral_tier=self.SBB)
        move_data_event_1 = cluster.move_data(self.env, self.ssd_tier, self.nvram_tier,
                                              total_volume=3e9,
                                              erase=False, data=self.data)  # SSD(200) -> NVRAM(400)
        move_data_event_2 = cluster.move_data(self.env, self.ssd_tier, self.nvram_tier,
                                              total_volume=3e9,
                                              erase=False, data=self.data)  # SSD(200) -> NVRAM(400)

        self.env.process(move_data_event_1)
        self.env.process(move_data_event_2)
        self.env.run()
        last_item = self.data.items[-1]
        self.assertEqual(last_item["bandwidth"], 100)
        self.assertEqual(last_item["tier_level"]["SSD"], 3e9)
        self.assertEqual(last_item["tier_level"]["NVRAM"], 6e9)

    def test_move_data_concurrency_1_without_erase(self):
        """Test that for simple data movement the retained bandwidth is the bottelneck of read and write"""
        self.nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
                                'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
                         'write': {'seq': 100, 'rand': 100}}
        hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                         'write': {'seq': 40, 'rand': 40}}

        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier, self.nvram_tier],
                          ephemeral_tier=self.SBB)
        move_data_event_1 = cluster.move_data(self.env, self.ssd_tier, self.nvram_tier,
                                              total_volume=3e9,
                                              erase=True, data=self.data)  # SSD(200) -> NVRAM(400)
        move_data_event_2 = cluster.move_data(self.env, self.hdd_tier, self.nvram_tier,
                                              total_volume=5e9,
                                              erase=True, data=self.data)  # HDD(80) -> NVRAM(400)
        self.env.process(move_data_event_1)
        self.env.process(move_data_event_2)
        self.env.run()
        last_item = self.data.items[-1]
        self.assertEqual(last_item["tier_level"]["HDD"], 0)
        self.assertEqual(last_item["tier_level"]["SSD"], 0)
        self.assertEqual(last_item["tier_level"]["NVRAM"], 8e9)


class TestClusterEphemeralService(unittest.TestCase):
    """Test data movement using ephemeral services as cache level"""

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

        self.SBB = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                                 bandwidth=self.nvram_bandwidth, capacity=10e9)
        self.cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                               ephemeral_tier=self.SBB)

    def test_move_data_to_bb(self):
        """Test moving data to burst buffer."""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier, self.nvram_tier],
                          ephemeral_tier=self.SBB)
        move_data_event = cluster.move_data(self.env, self.ssd_tier, self.SBB,
                                            total_volume=10e9,
                                            erase=False, data=self.data)  # SSD(200) -> NVRAM(400)

        self.env.process(move_data_event)
        self.env.run()
        last_item = self.data.items[-1]
        self.assertEqual(last_item["tier_level"]["SSD"], 10e9)
        self.assertEqual(last_item["BB_level"], 10e9)

    def test_destage_data_from_bb_no_erase(self):
        """Test reaching 90% of BB capacity and eviction policy."""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier, self.nvram_tier],
                          ephemeral_tier=self.SBB)
        # fill some data to SBB
        move_data_event = cluster.move_data(self.env, self.ssd_tier, self.SBB,
                                            total_volume=5e9,
                                            erase=False, data=self.data)  # SSD(200) -> NVRAM(400)

        destage_data_event = cluster.destage(self.env, self.SBB, self.ssd_tier,
                                             total_volume=5e9, erase=False, data=self.data)  # SSD(200) -> NVRAM(400
        self.env.process(move_data_event)
        self.env.process(destage_data_event)
        self.env.run()
        last_item = self.data.items[-1]
        # self.assertEqual(last_item["tier_level"]["SSD"], 10e9)
        self.assertEqual(last_item["BB_level"], 5e9)

    def test_destage_data_from_bb_with_erase(self):
        """Test reaching 90% of BB capacity and eviction policy."""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier, self.nvram_tier],
                          ephemeral_tier=self.SBB)
        # fill some data to SBB
        move_data_event = cluster.move_data(self.env, self.ssd_tier, self.SBB,
                                            total_volume=9.5e9,
                                            erase=False, data=self.data)  # SSD(200) -> NVRAM(400)

        destage_data_event = cluster.destage(self.env, self.SBB, self.ssd_tier,
                                             total_volume=5e9, erase=True, data=self.data)  # SSD(200) -> NVRAM(400
        self.env.process(move_data_event)
        self.env.process(destage_data_event)
        self.env.run()
        #last_item = self.data.items[-1]
        # self.assertEqual(last_item["tier_level"]["SSD"], 10e9)
        #self.assertEqual(last_item["BB_level"], 5e9)
