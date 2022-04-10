import simpy
from loguru import logger
import numpy as np
import pandas as pd
import math
from cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
import random
import string

"""TODO LIST:

            [OK] add start_delay as app parameter
            [OK] rename app.run(tiers <- placement)
            [OK] keep self.store internal
            [OK] superimpose two apps
            [OK] add id or name for each app and spread it in logs
"""


def monitor(data, lst):
    if isinstance(data, simpy.Store):
        data.put(lst)


def name_app():
    return ''.join(random.sample(string.ascii_uppercase, 1)) + str(random.randint(0, 9))


class Delay:
    def __init__(self, duration, data=None, appname=None):
        self.duration = duration
        self.data = data if data else None
        self.appname = appname if appname else ''

    def run(self, env, cluster):
        logger.info(f"(App {self.appname}) - Start waiting phase at {env.now}")
        t_start = env.now
        yield env.timeout(self.duration)
        t_end = env.now
        monitor(self.data,
                {"app": self.appname, "type": "wait", "cpu_usage": 0,
                 "t_start": t_start, "t_end": t_end, "bandwidth": 0,
                 "phase_duration": self.duration, "volume": 0,
                 "tiers": [tier.name for tier in cluster.tiers],
                 "data_placement": None,
                 "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}})
        logger.info(f"(App {self.appname}) - End waiting phase at {env.now}")
        return True


class IO_Compute:
    def __init__(self, duration, cores=1, data=None, appname=None):
        self.duration = duration
        self.cores = cores
        self.data = data if data else None
        self.appname = appname if appname else ''

    def run(self, env, cluster):
        used_cores = []
        # use self.cores
        for i in range(self.cores):
            core = cluster.compute_cores.request()
            used_cores.append(core)
            yield core
        logger.info(f"(App {self.appname}) - Start computing phase at {env.now} with {self.cores} requested cores")
        phase_duration = self.duration/compute_share_model(self.cores)
        # phase_duration = self.duration/compute_share_model(cluster.compute_cores.capacity - cluster.compute_cores.count)
        t_start = env.now
        yield env.timeout(phase_duration)

        t_end = env.now
        monitor(self.data,
                {"app": self.appname, "type": "compute", "cpu_usage": self.cores,
                 "t_start": t_start, "t_end": t_end, "bandwidth": 0,
                 "phase_duration": phase_duration, "volume": 0,
                 "tiers": [tier.name for tier in cluster.tiers],
                 "data_placement": None,
                 "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}})

        for core in used_cores:
            cluster.compute_cores.release(core)

        logger.info(f"(App {self.appname}) - End computing phase at {env.now} and releasing {self.cores} cores")
        return True


class IO_Phase:
    def __init__(self, operation='read', volume=1e9, pattern=1, data=None, appname=None):
        """
        pattern = 0.8:
            80% sequential and 20% random.
            blocksize for moment is not variable."""
        self.operation = operation
        assert self.operation in ['read', 'write']
        self.volume = volume
        self.pattern = pattern
        self.data = data if data else None
        self.appname = appname if appname else ''
        # logger.info(self.__str__())

    def __str__(self):
        io_pattern = f"{self.pattern*100}% sequential | {(1-self.pattern)*100} % random"
        description = "-------------------\n"
        description += (f"{self.operation.capitalize()} I/O Phase of volume {convert_size(self.volume)} with pattern: {io_pattern}\n")
        return description

    def update_tiers(self, cluster, tier):
        tier = get_tier(tier, cluster)
        # reading operation suppose at least some volume in the tier
        if self.operation == "read" and tier.capacity.level < self.volume:
            tier.capacity.put(self.volume)
        if self.operation == "write":
            tier.capacity.put(self.volume)

    def run(self, env, cluster, cores=1, placement=None):
        # Pre compute parameters
        tier = get_tier(placement, cluster)
        bandwidth = (tier.bandwidth[self.operation]['seq'] * self.pattern + tier.bandwidth[self.operation]['rand']*(1-self.pattern)) * compute_share_model(cores)

        used_cores = []
        for i in range(cores):
            core = cluster.compute_cores.request()
            used_cores.append(core)
            yield core
        logger.info(f"(App {self.appname}) - Start {self.operation.capitalize()} I/O Phase with volume = {convert_size(self.volume)} at {env.now}")
        logger.info(f"(App {self.appname}) - {self.operation.capitalize()}(ing) I/O with bandwidth = {bandwidth} MB/s")
        io_bandwidth = bandwidth*bandwidth_share_model(cluster.compute_cores.count)
        phase_duration = (self.volume/1e6)/io_bandwidth
        t_start = env.now
        self.update_tiers(cluster, tier)
        yield env.timeout(phase_duration)

        t_end = env.now
        monitor(self.data,
                {"app": self.appname, "type": self.operation, "cpu_usage": cores,
                 "t_start": t_start, "t_end": t_end, "bandwidth": io_bandwidth,
                 "phase_duration": phase_duration, "volume": self.volume,
                 "tiers": [tier.name for tier in cluster.tiers],
                 "data_placement": {"placement": tier.name},
                 "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}})
        # "tiers_level": [tier.capacity.level for tier in cluster.tiers]})
        for core in used_cores:
            cluster.compute_cores.release(core)

        logger.info(f"(App {self.appname}) - End {self.operation.capitalize()} I/O Phase at {env.now}")
        return True

        # with cluster.compute_cores.request() as req:
        #     yield req
        #     logger.info(f"Start I/O phase with {cluster.compute_cores.count} cores at {env.now}")
        #     speed_factor = speed_share_model(cluster.compute_cores.count)
        #     speed = cluster.storage_speed.level
        #     yield cluster.storage_speed.get(speed)
        #     yield cluster.storage_capacity.put(self.volume)
        #     yield env.timeout(self.volume/(speed*speed_factor))
        #     yield cluster.storage_speed.put(speed)
        #     logger.info(f"End I/O phase at {env.now}")


class Application:
    def __init__(self, env, name=None, compute=[0, 10], read=[1e9, 0], write=[0, 5e9], data=None, delay=0):
        self.env = env
        self.name = name if name else name_app()
        self.store = simpy.Store(self.env)
        self.compute = compute
        self.read = read
        self.write = write
        self.delay = delay
        # ensure format is valid, all list are length equal
        assert all([len(lst) == len(self.compute) for lst in [self.read, self.write]])
        self.data = data if data else None
        self.status = None
        # schedule all events
        self.schedule()

    def put_delay(self, duration):
        delay_phase = Delay(duration, data=self.data, appname=self.name)
        self.store.put(delay_phase)

    def put_compute(self, duration, cores=1):
        # self.env.process(run_compute_phase(cluster, self.env, duration, cores=cores))
        # store.put(run_compute_phase(cluster, self.env, duration, cores=cores))
        io_compute = IO_Compute(duration, cores, data=self.data, appname=self.name)
        self.store.put(io_compute)

    def put_io(self, operation, volume, pattern=1):
        # self.env.process(run_io_phase(cluster, self.env, volume))
        # store.put(run_io_phase(cluster, self.env, volume))
        io_phase = IO_Phase(operation=operation, volume=volume, pattern=pattern, data=self.data, appname=self.name)
        self.store.put(io_phase)

    def schedule(self):
        self.status = []
        if self.delay > 0:
            self.put_delay(duration=self.delay)
            self.status.append(False)
        for i in range(len(self.compute)):
            # read is prioritary
            if self.read[i] > 0:
                self.put_io(operation="read", volume=self.read[i])
                # read_io = IO_Phase(operation='read', volume=self.read[i])
                # self.store.put(read_io)
                self.status.append(False)
            # then write
            if self.write[i] > 0:
                self.put_io(operation="write", volume=self.write[i])
                # write_io = IO_Phase(operation='write', volume=self.write[i])
                # self.store.put(write_io)
                self.status.append(False)
            # then compute duration = diff between two events
            if i < len(self.compute) - 1:
                duration = self.compute[i+1] - self.compute[i]
                self.put_compute(duration, cores=1)
                self.status.append(False)

    def run(self, cluster, tiers):
        # assert len(cluster.tiers) == len(tiers)
        item_number = 0
        phase = 0
        while self.store.items:
            item = yield self.store.get()
            if isinstance(item, Delay) and phase == 0:
                self.status[phase] = yield self.env.process(item.run(self.env, cluster))
                phase += 1
            elif isinstance(item, IO_Compute):
                # compute phase
                if phase == 0:
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster))
                    phase += 1
                elif phase > 0 and self.status[phase-1] == True:
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster))
                    phase += 1
                else:
                    self.status[phase] = False
            else:
                print(f"item_number = {item_number} while tiers={tiers}")
                print(f"status list = {self.status}")
                placement = cluster.tiers[tiers[item_number]]
                if phase == 0:
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster, cores=1, placement=placement))
                    phase += 1
                elif phase > 0 and self.status[phase-1] == True:
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster, cores=1, placement=placement))
                    phase += 1
                else:
                    self.status[phase] = False
                    phase += 1
                item_number += 1
            # print(self.status)
        return self.data


    # def run(self, env, cluster):
if __name__ == '__main__':
    env = simpy.Environment()
    data = simpy.Store(env)
    # env.process(run_compute_phase(cluster, env, duration=10, cores=3))
    nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                       'write': {'seq': 515, 'rand': 505}}
    ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                     'write': {'seq': 100, 'rand': 100}}

    ssd_tier = Tier(env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
    nvram_tier = Tier(env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)
    cluster = Cluster(env,  compute_nodes=1, cores_per_node=2, tiers=[ssd_tier, nvram_tier])
    app1 = Application(env,
                       compute=[0, 10],
                       read=[1e9, 0],
                       write=[0, 5e9],
                       data=data)
    app2 = Application(env,
                       compute=[0],
                       read=[3e9],
                       write=[0],
                       data=data)

    # app2 = Application(env, store,
    #                    compute=[0, 25],
    #                    read=[2e9, 0],
    #                    write=[0, 10e9],
    #                    tiers=[0, 1])
    env.process(app1.run(cluster, tiers=[0, 0]))
    env.process(app2.run(cluster, tiers=[1, 1]))
    env.run()
    # print(cluster.compute_cores.capacity)
    # print(cluster.compute_cores.data)
    for item in data.items:
        print(item)
    # app.put_compute(duration=10, cores=2)
    # app.put_io(volume=2e9)
    # job.put_compute(duration=10, cores=2)
    # env.process(run_io_phase(cluster, env, 10e9))
    # env.process(app.run(cluster))
    # env.run(until=20)
