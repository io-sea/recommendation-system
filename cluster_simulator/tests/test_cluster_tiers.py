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
