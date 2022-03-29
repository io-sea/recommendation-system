import simpy
from loguru import logger
import numpy as np


class Cluster:
    def __init__(self, env, compute_cores=12, storage_capacity=1e10, storage_speed=2e9):
        self.compute_cores = simpy.Resource(env, capacity=compute_cores)
        self.storage_capacity = simpy.Container(env, init=0, capacity=storage_capacity)
        self.storage_speed = simpy.Container(env, init=storage_speed, capacity=storage_speed)


class ClusterT:
    def __init__(self, env, compute_cores=12, storage_capacity=1e10, storage_speed=2e9):
        self.compute_cores = simpy.Resource(env, capacity=compute_cores)
        self.storage_capacity = simpy.Container(env, init=0, capacity=storage_capacity)
        self.storage_speed = simpy.Container(env, init=storage_speed, capacity=storage_speed)


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
        #self.env.process(run_compute_phase(cluster, self.env, duration, cores=cores))
        #store.put(run_compute_phase(cluster, self.env, duration, cores=cores))
        io_compute = IO_Compute(duration, cores)
        self.store.put(io_compute)

    def put_io(self, volume):
        #self.env.process(run_io_phase(cluster, self.env, volume))
        #store.put(run_io_phase(cluster, self.env, volume))
        io_phase = IO_Phase(volume)
        self.store.put(io_phase)

    @property
    def get_store(self):
        return self.store

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
    #env.process(run_compute_phase(cluster, env, duration=10, cores=3))
    app = Application(env, store)
    # app.put_compute(duration=10, cores=2)
    # app.put_io(volume=2e9)
    #job.put_compute(duration=10, cores=2)
    # env.process(run_io_phase(cluster, env, 10e9))
    env.process(app.run(cluster))
    env.run(until=20)
