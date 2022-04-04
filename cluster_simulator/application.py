import simpy
from loguru import logger
import numpy as np
import pandas as pd
import math


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class Cluster:
    def __init__(self, env, compute_nodes=1, cores_per_node=2, tiers=[]):
        self.compute_nodes = simpy.Resource(env, capacity=compute_nodes)
        self.compute_cores = simpy.Resource(env, capacity=cores_per_node*compute_nodes)
        self.tiers = []
        for tier in tiers:
            if isinstance(tier, Tier):
                self.tiers.append(tier)
        logger.info(self.__str__())

    def __str__(self):
        description = "====================\n"
        description += (f"Cluster with {self.compute_nodes.capacity} compute nodes \n")
        description += f"Each having {self.compute_cores.capacity} cores in total \n"
        if self.tiers:
            for tier in self.tiers:
                description += tier.__str__()
        return description
        # self.storage_capacity = simpy.Container(env, init=0, capacity=storage_capacity)
        # self.storage_speed = simpy.Container(env, init=storage_speed, capacity=storage_speed)


class Tier:
    """
    In this model we expect a bandwidth value at its asymptotic state.
    Only the maximum is considered.
    Other considered variables are :
        read/write variables
        sequential/random variables
    Output is a scalar value in MB/s.
    Typically we access the bandwidth value as in dictionary: b['read']['seq'] = 200MB/s.
    TODO : extend this to a NN as function approximator to allow:
        averaging over variables
        interpolation when data entry is absent, i.e. b['seq'] gives a value
    """

    def __init__(self, env, name, bandwidth, capacity):
        self.name = name
        self.capacity = simpy.Container(env, init=0, capacity=capacity)
        self.bandwidth = bandwidth
        logger.info(self.__str__())

    def __str__(self):
        description = "-------------------\n"
        description += (f"Tier: {self.name} with capacity = {convert_size(self.capacity.capacity)}\n")
        description += ("{:<12} {:<12} {:<12}".format('Operation', 'Pattern', 'Bandwidth MB/s')+"\n")
        for op, inner_dict in self.bandwidth.items():
            for pattern, value in inner_dict.items():
                description += ("{:<12} {:<12} {:<12}".format(op, pattern, value)+"\n")
        return description


def speed_share_model(n_threads):
    return np.sqrt(1 + n_threads)/np.sqrt(2)


def compute_share_model(n_cores):
    return np.sqrt(1 + n_cores)/np.sqrt(2)


class IO_Compute:
    def __init__(self, duration, cores=1):
        self.duration = duration
        self.cores = cores

    def play(self, cluster, env):
        used_cores = []
        for i in range(self.cores):
            core = cluster.compute_cores.request()
            used_cores.append(core)
            yield core
        logger.info(f"Start computing phase at {env.now}")
        yield env.timeout(self.duration/compute_share_model(cluster.compute_cores.count))

        for core in used_cores:
            cluster.compute_cores.release(core)
        logger.info(f"End computing phase at {env.now}")


class IO_Phase:
    def __init__(self, volume, pattern=1):
        """
        pattern = 0.8:
            80% sequential and 20% random.
            blocksize for moment is not variable."""
        self.volume = volume
        self.pattern = pattern


class Read_IO_Phase(IO_Phase):
    def __init__(self, volume, pattern):
        super().__init__(volume, pattern)
        self.operation = 'read'
        logger.info(self.__str__())

    def __str__(self):
        io_pattern = f"{self.pattern*100}% sequential | {(1-self.pattern)*100} % random"
        description = "-------------------\n"
        description += (f"{self.operation.upper()} I/O Phase of volume {convert_size(self.volume)} with pattern: {io_pattern}\n")
        return description
        #self.read_bandwidth = tier.bandwidth['read']

    @staticmethod
    def get_tier(tier, cluster):
        if isinstance(tier, int):
            tier = cluster.tiers[tier]
        if isinstance(tier, str):
            for cluster_tier in cluster.tiers:
                if tier == cluster_tier.name:
                    tier = cluster_tier
        return tier

    def schedule(self, env, cluster, tier=None, cores=1):
        # Pre compute parameters
        tier = self.get_tier(tier, cluster)

        bandwidth = tier.bandwidth[self.operation]['seq'] * self.pattern + tier.bandwidth[self.operation]['rand']*(1-self.pattern) * compute_share_model(cores)

        used_cores = []
        for i in range(cores):
            core = cluster.compute_cores.request()
            used_cores.append(core)
            yield core
        logger.info(f"Start Read I/O Phase with volume={convert_size(self.volume)} at {env.now}")
        logger.info(f"Reading I/O with bandwidth = {bandwidth} MB/s")
        yield env.timeout((self.volume/1e6)/bandwidth)
        for core in used_cores:
            cluster.compute_cores.release(core)
        logger.info(f"End Read I/O Phase at {env.now}")

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
    def __init__(self, env, store, compute=[0, 10], read=[1e9, 0], write=[0, 5e9]):
        self.env = env
        self.store = store
        self.compute = compute
        self.read = read
        self.write = write
        # ensure format is valid, all list are length equal
        assert all([len(lst) == len(self.compute) for lst in [self.read, self.write]])
        # schedule all events
        self.schedule()

    def put_compute(self, duration, cores=1):
        # self.env.process(run_compute_phase(cluster, self.env, duration, cores=cores))
        # store.put(run_compute_phase(cluster, self.env, duration, cores=cores))
        io_compute = IO_Compute(duration, cores)
        self.store.put(io_compute)

    def put_io(self, volume):
        # self.env.process(run_io_phase(cluster, self.env, volume))
        # store.put(run_io_phase(cluster, self.env, volume))
        io_phase = IO_Phase(volume)
        self.store.put(io_phase)

    def schedule(self):
        for i in range(len(self.compute)):
            # read is prioritary
            if self.read[i] > 0:
                self.put_io(volume=self.read[i])
            # then write
            if self.write[i] > 0:
                self.put_io(volume=self.write[i])
            # then compute duration = diff between two events
            if i < len(self.compute) - 1:
                duration = self.compute[i+1] - self.compute[i]
                self.put_compute(duration)

    def run(self, cluster):
        while True:
            item = yield store.get()
            yield self.env.process(item.play(cluster, self.env))

    # def run(self, env, cluster):
if __name__ == '__main__':
    env = simpy.Environment()
    cluster = Cluster(env)
    store = simpy.Store(env, capacity=1000)
    # env.process(run_compute_phase(cluster, env, duration=10, cores=3))
    app = Application(env, store)
    # app.put_compute(duration=10, cores=2)
    # app.put_io(volume=2e9)
    # job.put_compute(duration=10, cores=2)
    # env.process(run_io_phase(cluster, env, 10e9))
    env.process(app.run(cluster))
    env.run(until=20)
