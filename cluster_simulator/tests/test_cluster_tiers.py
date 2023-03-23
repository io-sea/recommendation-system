import os
import unittest
import time
import numpy as np
import pandas as pd
import simpy
from loguru import logger

from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.application import Application

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_CONFIG_FILE = os.path.join(CURRENT_DIR, "test_data", "config.yaml")


class TestClusterConfigFile(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        # NOTE: in pyyaml, scientif. notation includes "." and "+/-"
        # example: 100e9 -> 100.0e+9
        # see https://github.com/yaml/pyyaml/issues/173

    def test_config_path(self):
        # Test that the cluster is initialized correctly using a YAML config file
        cluster = Cluster(self.env, config_path=TEST_CONFIG_FILE)
        print(cluster.compute_nodes)
        print(type(cluster.compute_nodes))
        self.assertEqual(cluster.compute_nodes.capacity, 1)
        self.assertEqual(cluster.compute_cores.capacity, 2)
        self.assertEqual(len(cluster.tiers), 3)
        self.assertEqual(cluster.ephemeral_tier.name, 'ephemeral')

    def test_override_compute_nodes_and_cores_per_node(self):
        # Test that the cluster is initialized correctly using overridden values for compute nodes and cores per node
        compute_nodes = 8
        cores_per_node = 16
        cluster = Cluster(self.env, config_path=TEST_CONFIG_FILE, compute_nodes=compute_nodes,
                          cores_per_node=cores_per_node)
        self.assertEqual(cluster.compute_nodes.capacity, compute_nodes)
        self.assertEqual(cluster.compute_cores.capacity, compute_nodes * cores_per_node)
        self.assertEqual(len(cluster.tiers), 3)
        self.assertEqual(cluster.ephemeral_tier.name, 'ephemeral')

    def test_override_tiers(self):
        # Test that the cluster is initialized correctly using overridden values for tiers
        tiers = [Tier(self.env, 'tier1', 100e9), Tier(self.env, 'tier2', 500e9)]
        cluster = Cluster(self.env, tiers=tiers)
        self.assertEqual(cluster.compute_nodes.capacity, 1)
        self.assertEqual(cluster.compute_cores.capacity, 2)
        self.assertEqual(len(cluster.tiers), 2)
        self.assertIsNone(cluster.ephemeral_tier)

    def test_override_ephemeral_tier(self):
        # Test that the cluster is initialized correctly using overridden values for the ephemeral tier
        TEST_CONFIG_FILE_NO_EPHEMERAL = os.path.join(CURRENT_DIR, "test_data", "config_no_ephemeral.yaml")
        ephemeral_tier = Tier(self.env, 'my_ephemeral', 200e9)
        cluster = Cluster(self.env, config_path=TEST_CONFIG_FILE_NO_EPHEMERAL, ephemeral_tier=ephemeral_tier)
        self.assertEqual(cluster.compute_nodes.capacity, 1)
        self.assertEqual(cluster.compute_cores.capacity, 2)
        self.assertEqual(len(cluster.tiers), 3)
        self.assertEqual(cluster.ephemeral_tier.name, 'my_ephemeral')

    def test_override_all_parameters(self):
        # Test that the cluster is initialized correctly using overridden values for all parameters
        compute_nodes = 8
        cores_per_node = 16
        tiers = [Tier(self.env, 'tier1', capacity=100e9), Tier(self.env, 'tier2', capacity=500e9)]
        ephemeral_tier = Tier(self.env, 'my_ephemeral', capacity=200e9)
        cluster = Cluster(self.env, config_path=TEST_CONFIG_FILE, compute_nodes=compute_nodes,
                          cores_per_node=cores_per_node, tiers=tiers,
                          ephemeral_tier=ephemeral_tier)
        self.assertEqual(cluster.compute_nodes.capacity, compute_nodes)
        self.assertEqual(cluster.compute_cores.capacity, compute_nodes * cores_per_node)
        self.assertEqual(len(cluster.tiers), 2)
        self.assertEqual(cluster.ephemeral_tier.name, 'my_ephemeral')


class TestClusterConfigFileBandwidth(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        TEST_CONFIG_FILE_WITH_MODEL = os.path.join(CURRENT_DIR, "test_data", "config_with_model.yaml")
        self.cluster = Cluster(self.env, config_path=TEST_CONFIG_FILE_WITH_MODEL)

    def test_config_path_with_model(self):
        # Test that the cluster is initialized correctly using a YAML config file
        self.assertEqual(self.cluster.tiers[0].capacity.capacity, 100e9)
        self.assertEqual(self.cluster.tiers[1].capacity.capacity, 500e9)
        self.assertEqual(self.cluster.tiers[2].capacity.capacity, 500e9)
        self.assertEqual(self.cluster.tiers[3].capacity.capacity, 500e9)
        self.assertEqual(self.cluster.ephemeral_tier.capacity.capacity, 50e9)

    def test_get_max_bandwidth(self):
        self.assertEqual(self.cluster.tiers[0].get_max_bandwidth(operation='read', pattern=1), 5)
        self.assertEqual(self.cluster.tiers[0].get_max_bandwidth(operation='write', pattern=0), 10)
        self.assertEqual(self.cluster.tiers[3].get_max_bandwidth(), 40)
        self.assertEqual(self.cluster.tiers[3].get_max_bandwidth(operation='write'), 40)
        self.assertEqual(self.cluster.tiers[3].get_max_bandwidth(pattern=0), 40)

    def test_get_max_bandwidth_with_model(self):
        new_data = pd.DataFrame({
            'nodes': [1, 1],
            'read_io_size': [8e6, 6e6],
            'write_io_size': [8e6, 6e6],
            'read_volume': [169e6, 200e6],
            'write_volume': [330e6, 200e6],
            'read_io_pattern': ['stride', 'seq'],
            'write_io_pattern': ['stride', 'seq'],
        })
        self.assertIsInstance(self.cluster.tiers[1].get_max_bandwidth(new_data=new_data), np.ndarray)

    def test_get_max_bandwidth_with_model_pure_read_write(self):
        new_data = pd.DataFrame({ #phase features
            'nodes': [1, 1],
            'read_io_size': [8e6, 6e6],
            'write_io_size': [8e6, 6e6],
            'read_volume': [169e6, 0],
            'write_volume': [0, 200e6],
            'read_io_pattern': ['stride', 'seq'],
            'write_io_pattern': ['stride', 'seq'],
        })
        self.assertIsInstance(self.cluster.tiers[1].get_max_bandwidth(new_data=new_data), np.ndarray)

class TestClusterConfigFileBandwidthApplication(unittest.TestCase):

    def setUp(self):
        self.env = simpy.Environment()
        TEST_CONFIG_FILE_WITH_MODEL = os.path.join(CURRENT_DIR, "test_data", "config_with_model.yaml")
        self.cluster = Cluster(self.env, config_path=TEST_CONFIG_FILE_WITH_MODEL)
    def test_get_max_bandwidth_with_model_with_application(self):
        data = simpy.Store(self.env)
        compute = [0,  10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [1, 0]
        app = Application(self.env,
                          compute=compute,
                          read=read,
                          write=write)
        self.env.process(app.run(self.cluster, placement=[0, 0]))
        self.env.run()

class TestClusterTiers(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
                                'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
                         'write': {'seq': 100, 'rand': 100}}
        hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                         'write': {'seq': 40, 'rand': 40}}
        self.hdd_tier = Tier(self.env, 'HDD', max_bandwidth=hdd_bandwidth, capacity=1e12)
        self.ssd_tier = Tier(self.env, 'SSD', max_bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', max_bandwidth=self.nvram_bandwidth, capacity=10e9)

    def test_get_tier_as_object_and_recursive(self):
        """Test reaching tier object using object"""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier])
        result = get_tier(cluster, tier_reference=self.hdd_tier)
        result2 = get_tier(cluster, result)
        self.assertIsInstance(result, Tier)
        self.assertEqual(result, self.hdd_tier)
        self.assertEqual(result2, self.hdd_tier)

    def test_get_tier_as_None(self):
        """Test reaching tier object using None as tier_reference generates Exception."""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier])
        with self.assertRaises(Exception):
            result = get_tier(cluster, tier_reference=None)

    def test_get_tier_as_int_reference(self):
        """Test reaching tier object using object"""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier])
        result = get_tier(cluster, tier_reference=1)
        self.assertEqual(result, self.ssd_tier)

    def test_get_tier_name_no_bb(self):
        """Test reaching tier object using integer or string"""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier])
        result = get_tier(cluster, tier_reference="HDD")
        self.assertEqual(result.name, "HDD")
        result = get_tier(cluster, 0)
        self.assertEqual(result.name, "HDD")

    def test_get_tier_name_with_bb(self):
        """Test reaching tier object using integer or string"""
        # define burst buffer with its backend PFS
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           max_bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        result = get_tier(cluster, tier_reference=0, use_bb=True)
        self.assertEqual(result.name, "BB")
        self.assertIsInstance(result, EphemeralTier)
        # retrieve bb backend tier
        self.assertEqual(result.persistent_tier, self.hdd_tier)

    def test_get_tier_with_bb(self):
        """Test reaching tier object using integer or string"""
        # define burst buffer with its backend PFS
        bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                           max_bandwidth=self.nvram_bandwidth, capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=bb)
        result = get_tier(cluster, tier_reference="BB", use_bb=True)
        self.assertEqual(result.name, "BB")
        self.assertIsInstance(result, EphemeralTier)
        # retrieve bb backend tier
        self.assertEqual(result.persistent_tier, self.hdd_tier)

    def test_create_transient_tier(self):
        """Test init and creation of a tier"""
        sbb_tier = EphemeralTier(self.env, name='SBB',
                                 persistent_tier=self.ssd_tier,
                                 max_bandwidth=self.nvram_bandwidth,
                                 capacity=10e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                          ephemeral_tier=sbb_tier)
        self.assertIsInstance(cluster.ephemeral_tier, EphemeralTier)


class TestClusterMonitoring(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.data = simpy.Store(self.env)
        self.nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
                                'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
                         'write': {'seq': 100, 'rand': 100}}
        hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                         'write': {'seq': 40, 'rand': 40}}
        self.hdd_tier = Tier(self.env, 'HDD', max_bandwidth=hdd_bandwidth, capacity=1e12)
        self.ssd_tier = Tier(self.env, 'SSD', max_bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', max_bandwidth=self.nvram_bandwidth,
                               capacity=10e9)
        self.bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                                max_bandwidth=self.nvram_bandwidth, capacity=10e9)

    def test_register_levels_empty(self):
        """Test registering tiers levels"""
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier, self.nvram_tier],
                          ephemeral_tier=self.bb)
        levels = cluster.get_levels()
        for (key, value) in levels.items():
            self.assertEqual(value, 0)

    def test_register_levels_filled(self):
        """Test registering tiers levels"""
        self.hdd_tier.capacity.put(1e9)
        self.ssd_tier.capacity.put(1e9)
        self.nvram_tier.capacity.put(1e9)
        self.bb.capacity.put(1e9)
        cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier, self.nvram_tier],
                          ephemeral_tier=self.bb)
        levels = cluster.get_levels()
        for (key, value) in levels.items():
            self.assertEqual(value, 1e9)


class TestClusterTierEviction(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        self.nvram_bandwidth = {'read':  {'seq': 800, 'rand': 800},
                                'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 200, 'rand': 200},
                         'write': {'seq': 100, 'rand': 100}}
        hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                         'write': {'seq': 40, 'rand': 40}}
        self.hdd_tier = Tier(self.env, 'HDD', max_bandwidth=hdd_bandwidth, capacity=1e12)
        self.ssd_tier = Tier(self.env, 'SSD', max_bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', max_bandwidth=self.nvram_bandwidth, capacity=10e9)
        self.bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                                max_bandwidth=self.nvram_bandwidth, capacity=10e9)
        self.cluster = Cluster(self.env, tiers=[self.hdd_tier, self.ssd_tier],
                               ephemeral_tier=self.bb)

    def test_evict_tier(self):
        """Test default evicting a tier"""
        # put data > upper threshold
        self.bb.capacity.put(9.5e9)
        self.bb.dirty = 5e9
        self.assertEqual(self.bb.capacity.level, 9.5e9)
        # should evict min(2.5e9, 4e9) = 2.5e9 -> to the lower level
        self.assertEqual(self.bb.evict(), 2.5e9)
        self.assertEqual(self.bb.capacity.level, 7e9)

    def test_evict_tier_low_level(self):
        """Test threshold not reached evicting a tier"""
        # put data < upper threshold
        self.bb.capacity.put(7.5e9)
        self.bb.dirty = 5e9
        self.assertEqual(self.bb.capacity.level, 7.5e9)
        self.assertEqual(self.bb.evict(), 0)
        self.assertEqual(self.bb.capacity.level, 7.5e9)

    def test_evict_tier_dirty_data(self):
        """Test that dirty data cannot be evicted."""
        # put data < upper threshold
        self.bb.capacity.put(9.5e9)
        self.bb.dirty = 9.5e9
        self.assertEqual(self.bb.capacity.level, 9.5e9)
        self.assertEqual(self.bb.evict(), 0)
        self.assertEqual(self.bb.capacity.level, 9.5e9)

    def test_evict_tier_partial_dirty_data(self):
        """Test that dirty data limit the volume to be evicted."""
        # put data < upper threshold
        self.bb.capacity.put(9.5e9)
        self.bb.dirty = 8e9
        self.assertEqual(self.bb.capacity.level, 9.5e9)
        # should evict min(2.5e9, 1e9) = 1e9
        self.assertEqual(self.bb.evict(), 1.5e9)
        self.assertEqual(self.bb.capacity.level, 8e9)

    def test_continuous_filling_and_eviction(self):
        """Test that the tier is filled and evicted continuously"""
        # put total_data > bb.capacity
        total_data = 15e9
        quantum = 1e9
        while total_data > 0:
            print(f"HDD_level : {convert_size(self.hdd_tier.capacity.level)} | BB_level : {convert_size(self.bb.capacity.level)} | BB_dirty : {convert_size(self.bb.dirty)}")

            # sending data to BB
            self.bb.capacity.put(quantum)
            self.bb.dirty += quantum
            total_data -= quantum
            # destaging data to HDD
            self.hdd_tier.capacity.put(quantum)
            self.bb.dirty -= quantum
            self.bb.evict()
