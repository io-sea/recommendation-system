import simpy
from loguru import logger
import numpy as np
"""
How to schedule events after others
https://simpy.readthedocs.io/en/latest/topical_guides/events.html
>>> class School:
...     def __init__(self, env):
...         self.env = env
...         self.class_ends = env.event()
...         self.pupil_procs = [env.process(self.pupil()) for i in range(3)]
...         self.bell_proc = env.process(self.bell())
...
...     def bell(self):
...         for i in range(2):
...             yield self.env.timeout(45)
...             self.class_ends.succeed()
...             self.class_ends = self.env.event()
...             print()
...
...     def pupil(self):
...         for i in range(2):
...             print(r' \o/', end='')
...             yield self.class_ends

"""


class Cluster:
    def __init__(self, env, compute_cores=12, storage_capacity=1e10, storage_speed=2e9):
        self.compute_cores = simpy.Resource(env, capacity=compute_cores)
        self.storage_capacity = simpy.Container(env, init=0, capacity=storage_capacity)
        self.storage_speed = simpy.Container(env, init=storage_speed, capacity=storage_speed)


def speed_share_model(n_threads):
    return np.sqrt(1 + n_threads)


def compute_share_model(n_cores):
    return np.sqrt(1 + n_cores)/np.sqrt(2)


def run_compute_phase(cluster, env, duration, cores=1):
    """running a compute phase by exploiting a compute node"""
    # with cluster.compute_cores.request() as req:
    #     yield req
    #     logger.info(f"Start computing phase at {env.now}")
    #     yield env.timeout(duration)
    #     logger.info(f"End computing phase at {env.now}")
    used_cores = []
    for i in range(cores):
        core = cluster.compute_cores.request()
        used_cores.append(core)
        yield core

    # core = cluster.compute_cores.request()
    # yield core
    logger.info(f"Start computing phase at {env.now}")
    logger.info(f"{cluster.compute_cores.count} are currently using HPC cores")
    yield env.timeout(duration/compute_share_model(cluster.compute_cores.count))

    # cluster.compute_cores.release(core)
    for core in used_cores:
        cluster.compute_cores.release(core)
    logger.info(f"{cluster.compute_cores.count} are currently using HPC cores")
    logger.info(f"End computing phase at {env.now}")


def run_io_phase(cluster, env, volume):
    """running an IO phase with volume
    TODO: add type R/W and identify each"""
    with cluster.compute_cores.request() as req:
        yield req
        logger.info(f"Start I/O phase with {cluster.compute_cores.count} cores at {env.now}")
        speed_factor = speed_share_model(cluster.compute_cores.count)
        speed = cluster.storage_speed.level
        yield cluster.storage_speed.get(speed)
        yield cluster.storage_capacity.put(volume)
        yield env.timeout(volume/(speed*speed_factor))
        yield cluster.storage_speed.put(speed)
        logger.info(f"End I/O phase at {env.now}")


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

        # core = cluster.compute_cores.request()
        # yield core
        logger.info(f"Start computing phase at {env.now}")
        logger.info(f"{cluster.compute_cores.count} are currently using HPC cores")
        yield env.timeout(self.duration/compute_share_model(cluster.compute_cores.count))

        # cluster.compute_cores.release(core)
        for core in used_cores:
            cluster.compute_cores.release(core)
        logger.info(f"{cluster.compute_cores.count} are currently using HPC cores")
        logger.info(f"End computing phase at {env.now}")


class IO_Phase:
    def __init__(self, volume):
        self.volume = volume

    def play(self, cluster, env):
        with cluster.compute_cores.request() as req:
            yield req
            logger.info(f"Start I/O phase with {cluster.compute_cores.count} cores at {env.now}")
            speed_factor = speed_share_model(cluster.compute_cores.count)
            speed = cluster.storage_speed.level
            yield cluster.storage_speed.get(speed)
            yield cluster.storage_capacity.put(self.volume)
            yield env.timeout(self.volume/(speed*speed_factor))
            yield cluster.storage_speed.put(speed)
            logger.info(f"End I/O phase at {env.now}")


class Job:
    def __init__(self, env, store):
        self.env = env
        self.store = simpy.Store(env)

    def put_compute(self, duration, cores=1):
        #self.env.process(run_compute_phase(cluster, self.env, duration, cores=cores))
        #store.put(run_compute_phase(cluster, self.env, duration, cores=cores))
        io_compute = IO_Compute(duration, cores)
        store.put(io_compute)

    def put_io(self, volume):
        #self.env.process(run_io_phase(cluster, self.env, volume))
        #store.put(run_io_phase(cluster, self.env, volume))
        io_phase = IO_Phase(volume)
        store.put(io_phase)

    def run(self, cluster):
        while True:
            item = yield store.get()
            self.env.process(item.play(cluster, self.env))


if __name__ == '__main__':
    env = simpy.Environment()
    cluster = Cluster(env)
    store = simpy.Store(env, capacity=1000)
    #env.process(run_compute_phase(cluster, env, duration=10, cores=3))
    job = Job(env, store)
    job.put_compute(duration=10, cores=3)
    job.put_io(volume=1e9)
    # env.process(run_io_phase(cluster, env, 10e9))
    env.process(job.run(cluster))
    env.run(until=20)
    # print(cluster.storage_speed.capacity)
    # print(cluster.storage_speed.level)
    # print(cluster.storage_speed.get_queue)
    # https://stackoverflow.com/questions/48738371/
    # https://simpy.readthedocs.io/en/latest/topical_guides/resources.html#res-type-container
    # simpy-requesting-multiple-nonspecific-resources-and-order-of-requests

    # env.process(car(env))
    # env.run(until=15)
# def run(cluster, env):
#     # Simulate the following events
#     # Compute time of 10, exploiting one compute node out of 2 in the cluster
#     # Writing 5e9 into the storage at full available speed
#     with cluster.compute_cores.request() as req:
#         yield req
#         print(f"Start computing phase at {env.now}")
#         yield env.timeout(10)
#         print(f"End computing phase at {env.now}")

#     with cluster.compute_cores.request() as req:
#         yield req
#         print(f"Start requesting bandwidth for I/O at {env.now}")
#         bandwidth_share = cluster.storage_speed.capacity
#         print(f"Exploiting bandwidth for I/O : {bandwidth_share} out of {cluster.storage_speed.capacity}")
#         yield cluster.storage_speed.get(bandwidth_share)
#         print(f"Start I/O writing phase at {env.now} for 5e9")
#         io_time = 5e9/bandwidth_share/1000
#         print(f"I/O writing phase duration is {io_time}")
#         yield cluster.storage_capacity.put(5e9)
#         yield env.timeout(io_time)
#         yield cluster.storage_speed.put(bandwidth_share)
#         print(f"Finish I/O writing phase at {env.now} for 5e9")
#         print(f"Storage State at {env.now} is {cluster.storage_capacity.level} out of {cluster.storage_capacity.capacity}")
