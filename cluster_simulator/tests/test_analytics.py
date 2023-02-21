import sys
import simpy
from loguru import logger
import time
import numpy as np
import pandas as pd
import unittest
from cluster_simulator.utils import convex_hull
from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.application import Application
from cluster_simulator.analytics import display_run, interpolate_signal_from_simulation
from cluster_simulator.ephemeral_placement import ClusterBlackBox
from cluster_simulator.analytics import *
import time
import numpy as np
import pandas as pd
import os

# logger.remove()
# logger.level('DEBUG')

class TestAnalyticsSignals(unittest.TestCase):
    def test_signal_interpolation(self):
        t_starts = [0, 0, 20]
        t_ends = [20, 20, 30]

        read_bw = [105, 0, 0]
        write_bw = [0, 50, 100]

        x = np.arange(np.min(t_starts), np.max(t_ends), 1)

        result_read = interpolate_signal_from_simulation(x, t_starts, t_ends, read_bw)
        result_write = interpolate_signal_from_simulation(x, t_starts, t_ends, write_bw)

        self.assertListEqual(result_read, [105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertListEqual(result_write, [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])



class TestAnalytics(unittest.TestCase):
    def setUp(self):
        # sim env an data
        self.env = simpy.Environment()
        self.data = simpy.Store(self.env)
        # tier perfs
        nvram_bandwidth = {'read':  {'seq': 800, 'rand': 600},
                                'write': {'seq': 400, 'rand': 400}}
        ssd_bandwidth = {'read':  {'seq': 240, 'rand': 180},
                            'write': {'seq': 100, 'rand': 100}}
        hdd_bandwidth = {'read':  {'seq': 80, 'rand': 80},
                            'write': {'seq': 40, 'rand': 40}}

        # registering Tiers
        hdd_tier = Tier(self.env, 'HDD', bandwidth=hdd_bandwidth, capacity=1e12)
        hdd2_tier = Tier(self.env, 'HDD', bandwidth=hdd_bandwidth, capacity=1e12)
        ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth,
                                capacity=10e9)
        # registering Ephemeral Tier
        bb = EphemeralTier(self.env, name="BB", persistent_tier=ssd_tier,
                                bandwidth=nvram_bandwidth, capacity=10e9)

        # Define the cluster with 1 persistent and 1 ephemeral
        self.cluster = Cluster(self.env, compute_nodes=3, cores_per_node=2,
                          tiers=[hdd_tier, ssd_tier], ephemeral_tier=bb)

    def test_rw_mix(self):
        """Tests the display analytics for an application mixing read and write phases"""
        # Simple app: read 1GB -> compute 10s -> write 5GB
        read = [3e9, 2e9]
        compute = [0,  15]
        write = [0, 5e9]
        # placement
        placement = [0, 0]
        use_bb = [False, False]
        # simulate the app execution

        app1 = Application(self.env, name="app1",
                           compute=compute, read=read, write=write,
                           data=self.data)
        self.env.process(app1.run(self.cluster, placement=placement, use_bb=use_bb))
        self.env.run()
        print(f"application duration = {app1.get_fitness()}")

        fig = display_run(self.data, self.cluster, width=800, height=900)
        fig.show()


    def test_rw_mix_with_bb(self):

        """Tests the display analytics for an application mixing read and write phases with burst buffering"""
        # Simple app: read 1GB -> compute 10s -> write 5GB
        read = [3e9, 2e9]
        compute = [0,  30]
        write = [0, 5e9]
        # placement
        placement = [1, 1]
        use_bb = [True, True]
        # simulate the app execution

        app1 = Application(self.env, name="app1BB",
                           compute=compute, read=read, write=write,
                           data=self.data)
        self.env.process(app1.run(self.cluster, placement=placement, use_bb=use_bb))
        self.env.run()
        print(f"application duration = {app1.get_fitness()}")

        fig = display_run(self.data, self.cluster, width=800, height=900)
        fig.show()


    def test_rw_mix_with_two_apps(self):
        """Tests the display analytics for two applications mixing read and write phases"""

        # placement
        placement = [0, 0]
        use_bb = [False, False]
        # simulate the app execution

        app1 = Application(self.env, name="app#1",
                           compute=[0, 10], read=[1e9, 0], write=[0, 5e9],
                           data=self.data)
        app2 = Application(self.env, name="app#2",
                           compute=[0, 25],  read=[2e9, 0], write=[7e9, 0],
                           data=self.data)

        self.env.process(app1.run(self.cluster, placement=placement, use_bb=use_bb))
        self.env.process(app2.run(self.cluster, placement=placement, use_bb=use_bb))
        self.env.run()
        print(f"application duration = {app1.get_fitness()}")

        fig = display_run(self.data, self.cluster, width=800, height=900)
        fig.show()

    def test_rw_mix_with_many_apps_with_bb(self):
        """Tests the display analytics for two applications mixing read and write phases and using burst buffering."""

        # placement
        placement = [0, 0]
        # simulate the app execution

        app1 = Application(self.env,
                           compute=[0, 10], read=[1e9, 0], write=[0, 5e9],
                           data=self.data)
        app2 = Application(self.env,
                           compute=[0, 25],  read=[2e9, 0], write=[7e9, 0],
                           data=self.data)

        self.env.process(app1.run(self.cluster, placement=placement, use_bb=[False, True]))
        self.env.process(app2.run(self.cluster, placement=placement, use_bb=[False, False]))
        self.env.run()

        print(f"application duration = {app1.get_fitness()}")

        fig = display_run(self.data, self.cluster, width=800, height=900)
        fig.show()

    def test_rw_mix_with_many_apps_with_bb_last(self):
        """Tests the display analytics for two applications mixing read and write phases in first events and using burst buffering."""
        # placement
        placement = [1, 1]
        # simulate the app execution

        app1 = Application(self.env, name="app1BB",
                           compute=[0, 10], read=[1e9, 0], write=[0, 5e9],
                           data=self.data)
        app2 = Application(self.env, name="app2BB",
                           compute=[0, 25],  read=[0, 2e9], write=[7e9, 0],
                           data=self.data)

        self.env.process(app1.run(self.cluster, placement=placement, use_bb=[True, False]))
        self.env.process(app2.run(self.cluster, placement=placement, use_bb=[True, False]))
        self.env.run()

        print(f"application duration = {app1.get_fitness()}")

        fig = display_run(self.data, self.cluster, width=800, height=900)
        fig.show()

    def test_rw_mix_with_many_apps_with_bb_last(self):
        """Tests the display analytics for two applications mixing read and write phases in last events and using burst buffering."""

        # placement
        placement = [1, 1]
        # simulate the app execution

        app1 = Application(self.env, name="app1BB",
                           compute=[0, 10], read=[1e9, 0], write=[0, 5e9],
                           data=self.data)
        app2 = Application(self.env, name="app2BB",
                           compute=[0, 25],  read=[0, 2e9], write=[7e9, 0],
                           data=self.data)

        self.env.process(app1.run(self.cluster, placement=placement, use_bb=[False, True]))
        self.env.process(app2.run(self.cluster, placement=placement, use_bb=[False, True]))
        self.env.run()

        print(f"application duration = {app1.get_fitness()}")

        fig = display_run(self.data, self.cluster, width=800, height=900)
        fig.show()

if __name__ == '__main__':
    unittest.main(verbosity=2)
