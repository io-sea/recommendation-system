import simpy
from loguru import logger
import numpy as np
import pandas as pd
import math
from cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size


class IO_Compute:
    def __init__(self, duration, requested_cores=1):
        self.duration = duration
        self.requested_cores = requested_cores

    def run(self, env, cluster):
        cluster.compute_cores.get(self.requested_cores)
        logger.info(f"Start computing phase at {env.now} with {self.requested_cores} requested cores")
        yield env.timeout(self.duration/compute_share_model(cluster.compute_cores.capacity - cluster.compute_cores.level))
        cluster.compute_cores.put(self.requested_cores)
        logger.info(f"End computing phase at {env.now} and releasing {self.requested_cores} cores")
        return True


class IO_Phase:
    def __init__(self, operation='read', volume=1e9, pattern=1):
        """
        pattern = 0.8:
            80% sequential and 20% random.
            blocksize for moment is not variable."""
        self.operation = operation
        assert self.operation in ['read', 'write']
        self.volume = volume
        self.pattern = pattern
        # logger.info(self.__str__())

    def __str__(self):
        io_pattern = f"{self.pattern*100}% sequential | {(1-self.pattern)*100} % random"
        description = "-------------------\n"
        description += (f"{self.operation.capitalize()} I/O Phase of volume {convert_size(self.volume)} with pattern: {io_pattern}\n")
        return description

    def run(self, env, cluster, cores=1, tier=None):
        # Pre compute parameters
        tier = get_tier(tier, cluster)
        bandwidth = tier.bandwidth[self.operation]['seq'] * self.pattern + tier.bandwidth[self.operation]['rand']*(1-self.pattern) * compute_share_model(cores)

        cluster.compute_cores.get(cores)
        logger.info(f"Start {self.operation.capitalize()} I/O Phase with volume = {convert_size(self.volume)} at {env.now}")
        logger.info(f"{self.operation.capitalize()}(ing) I/O with bandwidth = {bandwidth} MB/s")
        speed_factor = bandwidth_share_model(cluster.compute_cores.level)
        yield env.timeout((self.volume/1e6)/(bandwidth*speed_factor))
        cluster.compute_cores.put(cores)
        logger.info(f"End {self.operation.capitalize()} I/O Phase at {env.now}")
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
    def __init__(self, env, store, compute=[0, 10], read=[1e9, 0], write=[0, 5e9], tiers=[0, 0]):
        self.env = env
        self.store = store
        self.compute = compute
        self.read = read
        self.write = write
        self.tiers = tiers
        # ensure format is valid, all list are length equal
        assert all([len(lst) == len(self.compute) for lst in [self.read, self.write, self.tiers]])
        # schedule all events
        self.status = None
        self.schedule()

    def put_compute(self, duration, cores=1):
        # self.env.process(run_compute_phase(cluster, self.env, duration, cores=cores))
        # store.put(run_compute_phase(cluster, self.env, duration, cores=cores))
        io_compute = IO_Compute(duration, cores)
        self.store.put(io_compute)

    def put_io(self, operation, volume, pattern=1):
        # self.env.process(run_io_phase(cluster, self.env, volume))
        # store.put(run_io_phase(cluster, self.env, volume))
        io_phase = IO_Phase(operation=operation, volume=volume, pattern=pattern)
        self.store.put(io_phase)

    def schedule(self):
        self.status = []
        for i in range(len(self.compute)):
            # read is prioritary
            if self.read[i] > 0:
                read_io = IO_Phase(operation='read', volume=self.read[i])
                self.store.put(read_io)
                self.status.append(False)
            # then write
            if self.write[i] > 0:
                write_io = IO_Phase(operation='write', volume=self.write[i])
                self.store.put(write_io)
                self.status.append(False)
            # then compute duration = diff between two events
            if i < len(self.compute) - 1:
                duration = self.compute[i+1] - self.compute[i]
                self.put_compute(duration, cores=3)
                self.status.append(False)

    def run(self, cluster):
        # TODO previous_event
        item_number = 0
        phase = 0
        while self.store.items:
            item = yield self.store.get()
            print(self.status)
            if isinstance(item, IO_Compute):
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
                tier = cluster.tiers[self.tiers[item_number]]
                if phase == 0:
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster, cores=1, tier=tier))
                    phase += 1
                elif phase > 0 and self.status[phase-1] == True:
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster, cores=1, tier=tier))
                    phase += 1
                else:
                    self.status[phase] = False
                #assert isinstance(tier, Tier)
                # yield self.env.process(item.run(self.env, cluster, cores=1, tier=tier))
                item_number += 1

        print(self.status)

    # def run(self, env, cluster):
if __name__ == '__main__':
    env = simpy.Environment()
    store = simpy.Store(env)
    # env.process(run_compute_phase(cluster, env, duration=10, cores=3))
    compute = [0, 10]
    read = [1e9, 0]
    write = [0, 5e9]
    tiers = [0, 1]
    nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
                       'write': {'seq': 515, 'rand': 505}}
    ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                     'write': {'seq': 100, 'rand': 100}}

    ssd_tier = Tier(env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
    nvram_tier = Tier(env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)
    cluster = Cluster(env,  compute_nodes=1, cores_per_node=4, tiers=[ssd_tier, nvram_tier])
    app = Application(env, store,
                      compute=compute,
                      read=read,
                      write=write,
                      tiers=tiers)
    env.process(app.run(cluster))
    app.env.run()
    print(cluster.compute_cores.capacity)
    print(cluster.compute_cores.data)
    # app.put_compute(duration=10, cores=2)
    # app.put_io(volume=2e9)
    # job.put_compute(duration=10, cores=2)
    # env.process(run_io_phase(cluster, env, 10e9))
    # env.process(app.run(cluster))
    # env.run(until=20)
