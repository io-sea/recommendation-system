import unittest
import time
import numpy as np
import simpy

from cluster_simulator.cluster import Cluster, Tier, EphemeralTier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
from cluster_simulator.application import Application
from analytics import display_run


class TestAppInit(unittest.TestCase):
    def setUp(self):
        self.env = simpy.Environment()
        nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                           'write': {'seq': 515, 'rand': 505}}
        ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                         'write': {'seq': 100, 'rand': 100}}

        self.ssd_tier = Tier(self.env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)

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
        compute = [0,  10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [1, 0]
        app = Application(self.env,
                          compute=compute,
                          read=read,
                          write=write)

        self.env.process(app.run(cluster, placement=tiers))
        self.env.run()
        # self.env.run(until=25)
        # self.assertEqual(len(app.store.items), 3)

    def test_app_fitness_no_data(self):
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

        self.env.process(app.run(cluster, placement=tiers))
        self.env.run()
        self.assertIsNone(app.get_fitness())

    def test_app_fitness(self):
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        # record data
        data = simpy.Store(self.env)
        # Simple app: read 1GB -> compute 10s -> write 5GB
        compute = [0, 10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [0, 1]
        app = Application(self.env,
                          compute=compute,
                          read=read,
                          write=write,
                          data=data)
        self.env.process(app.run(cluster, placement=tiers))
        self.env.run()
        self.assertAlmostEqual(app.get_fitness(), 24, places=0)

    def test_app_fitness_filter_name(self):
        """Tests that app fitness routine filters by the specified application name."""
        cluster = Cluster(self.env, tiers=[self.ssd_tier, self.nvram_tier])
        # record data
        data = simpy.Store(self.env)
        # Simple app: read 1GB -> compute 10s -> write 5GB
        compute = [0, 10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [0, 1]
        app = Application(self.env,
                          name="appname",
                          compute=compute,
                          read=read,
                          write=write,
                          data=data)
        self.env.process(app.run(cluster, placement=tiers))
        self.env.run()
        # fig = display_run(data, cluster, width=800, height=900)
        # fig.show()
        # self.assertAlmostEqual(app.get_fitness(app_name_filter="appname"), 24, places=0)
        # self.assertEqual(app.get_fitness(app_name_filter="app_name"), 0)


class TestBasicApps(unittest.TestCase):
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
        self.env.process(app1.run(cluster, placement=[0, 0]))
        self.env.run()
        for item in data.items:
            print(item)

    def test_app_rerun(self):
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0, 10],
                           read=[1e9, 0], write=[0, 5e9], data=data)
        self.env.process(app1.run(cluster, placement=[0, 0]))
        self.env.run()
        print(app1.get_fitness())
        print(data.items)
        self.env.process(app1.run(cluster, placement=[1, 1]))
        self.env.run()
        print(app1.get_fitness())
        print(data.items)

    def test_app_pure_read(self):
        data = simpy.Store(self.env)
        read_size = 1e9
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0],
                           read=[read_size], write=[0], data=data)

        self.env.process(app1.run(cluster, placement=[0]))
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
        self.env.process(app1.run(cluster, placement=[0]))
        self.env.run()
        self.assertEqual(len(data.items), 1)
        tier_name = data.items[0]["data_placement"]["placement"]
        self.assertEqual(data.items[0]["tier_level"][tier_name], write_size)

    def test_delayed_app(self):
        data = simpy.Store(self.env)
        write_size = 1e9
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0, 10],
                           read=[0, 0], write=[0, 0], data=data, delay=5)
        self.env.process(app1.run(cluster, placement=[0]))
        self.env.run()
        for item in data.items:
            print(item)
        self.assertEqual(data.items[0]["phase_duration"], data.items[1]["t_start"])


class TestPhaseSuperposition(unittest.TestCase):
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
        self.env.process(app1.run(cluster, placement=[0, 0]))
        self.env.process(app2.run(cluster, placement=[0, 0]))
        self.env.run()
        timespan = 0.0
        for item in data.items:
            print(item)
            timespan = max(timespan, item["t_end"])
        self.assertEqual(data.items[0]["t_end"], data.items[1]["t_start"])
        self.assertEqual(timespan, 25.0)

    def test_2_computes_parallel(self):
        """Two compute phases that should run in parallel because the cluster
        has now 2 cores."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, compute=[0, 15],
                           read=[0, 0], write=[0, 0], data=data)
        app2 = Application(self.env, compute=[0, 1],
                           read=[0, 0], write=[0, 0], data=data)
        self.env.process(app1.run(cluster, placement=[0, 0]))
        self.env.process(app2.run(cluster, placement=[0, 0]))
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
        self.env.process(app1.run(cluster, placement=[0, 0]))
        self.env.process(app2.run(cluster, placement=[0, 0]))
        self.env.run()
        self.assertEqual(data.items[0]["t_end"], data.items[1]["t_start"])

    def test_2_IO_parallel(self):
        """Two I/O phases that should run in parallel because the cluster
        has2 cores."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, name="#1", compute=[0],
                           read=[1e9], write=[0], data=data)
        app2 = Application(self.env, name="#2", compute=[0, 1],
                           read=[0, 0], write=[0, 2e9], data=data)
        self.env.process(app1.run(cluster, placement=[0, 0]))
        self.env.process(app2.run(cluster, placement=[0, 0]))
        self.env.run()

        self.assertEqual(data.items[0]["t_start"], data.items[1]["t_start"])

    def test_2_IO_parallel_1(self):
        """Two I/O phases that should run in parallel because the cluster
        has2 cores."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=2,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, name="#read2G->Comp2s", compute=[0, 2],
                           read=[1e9, 0], write=[0, 0], data=data)
        app2 = Application(self.env, name="#comp1s->write2G", compute=[0, 1],
                           read=[0, 0], write=[0, 2e9], data=data)
        self.env.process(app2.run(cluster, placement=[0, 0]))
        self.env.process(app1.run(cluster, placement=[0, 0]))

        self.env.run()
        # fig = display_run(data, cluster, width=800, height=900)
        # fig.show()
        self.assertEqual(data.items[0]["t_start"], data.items[1]["t_start"])

    def test_2_IO_parallel_2(self):
        """Two I/O phases that should run in parallel because the cluster
        has2 cores."""
        data = simpy.Store(self.env)
        cluster = Cluster(self.env,  compute_nodes=1, cores_per_node=3,
                          tiers=[self.ssd_tier, self.nvram_tier])
        app1 = Application(self.env, name="read3G->comp15s", compute=[0, 15], read=[3e9, 0],
                           write=[0, 0], data=data)
        app2 = Application(self.env, name="read1G->comp10s", compute=[0, 10], read=[1e9, 0],
                           write=[0, 0], data=data)

        self.env.process(app1.run(cluster, placement=[1, 1]))
        self.env.process(app2.run(cluster, placement=[1, 1]))
        self.env.run()
        # self.assertEqual(data.items[0]["t_start"], data.items[1]["t_start"])
        # fig = display_run(data, cluster, width=800, height=900)
        # fig.show()


class TestBufferedApplications(unittest.TestCase):
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
        self.nvram_tier = Tier(self.env, 'NVRAM', bandwidth=self.nvram_bandwidth,
                               capacity=10e9)
        self.bb = EphemeralTier(self.env, name="BB", persistent_tier=self.hdd_tier,
                                bandwidth=self.nvram_bandwidth, capacity=10e9)

    def test_noSBB_app(self):
        """Test running simple apps in cluster having a datanode with BB but not requested."""
        cluster = Cluster(self.env, tiers=[self.hdd_tier],
                          ephemeral_tier=self.bb)
        # Simple app: read 1GB -> compute 10s -> write 5GB
        compute = [0,  10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [0, 0]
        use_bb = [False, False]
        app = Application(self.env,
                          compute=compute,
                          read=read,
                          write=write,
                          data=self.data)
        self.env.process(app.run(cluster, placement=tiers, use_bb=use_bb))
        self.env.run()
        self.assertEqual(self.data.items[0]["type"], "read")
        self.assertEqual(self.data.items[1]["type"], "compute")
        self.assertEqual(self.data.items[2]["type"], "write")

    def test_SBB_app_write_phase(self):
        """Test running simple apps in cluster having a datanode with BB and a write operation"""
        cluster = Cluster(self.env, tiers=[self.hdd_tier],
                          ephemeral_tier=self.bb)
        # Simple app: read 1GB -> compute 10s -> write 5GB
        compute = [0,  10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [0, 0]
        use_bb = [False, True]
        app = Application(self.env,
                          compute=compute,
                          read=read,
                          write=write,
                          data=self.data)
        self.env.process(app.run(cluster, placement=tiers, use_bb=use_bb))
        self.env.run()
        self.assertEqual(self.data.items[0]["type"], "read")
        self.assertEqual(self.data.items[1]["type"], "compute")
        self.assertEqual(self.data.items[2]["type"], "write")
        self.assertEqual(self.data.items[3]["type"], "movement")

    def test_prefetch_SBB_app_read_phase(self):
        """Test running simple apps in cluster having a datanode with BB and a prefetch operation """
        cluster = Cluster(self.env, tiers=[self.hdd_tier],
                          ephemeral_tier=self.bb)
        # Simple app: read 1GB -> compute 10s -> write 5GB
        compute = [0,  10]
        read = [1e9, 0]
        write = [0, 5e9]
        tiers = [0, 0]
        use_bb = [True, False]
        app = Application(self.env,
                          compute=compute,
                          read=read,
                          write=write,
                          data=self.data)
        self.env.process(app.run(cluster, placement=tiers, use_bb=use_bb))
        self.env.run()
        self.assertEqual(self.data.items[0]["type"], "movement")
        self.assertEqual(self.data.items[1]["type"], "read")
        self.assertEqual(self.data.items[2]["type"], "compute")
        self.assertEqual(self.data.items[3]["type"], "write")

    def test_SBB_apps_with_concurrency(self):
        "Test running multiples apps concurrent in a single SBB."
        cluster = Cluster(self.env, tiers=[self.hdd_tier],
                          ephemeral_tier=self.bb)
        app1 = Application(self.env,
                           read=[2e9, 0],
                           compute=[0, 10],
                           write=[0, 5e9],
                           data=self.data, delay=0)
        app2 = Application(self.env,
                           read=[1e9, 0],
                           compute=[0, 6],
                           write=[0, 5e9],
                           data=self.data, delay=0)
        self.env.process(app1.run(cluster, placement=[0, 0],
                                  use_bb=[False, True]))
        self.env.process(app2.run(cluster, placement=[0, 0],
                                  use_bb=[False, True]))

        self.env.run()
        fig = display_run(self.data, cluster, width=800, height=900)
        fig.show()


if __name__ == '__main__':
    unittest.main(verbosity=2)
