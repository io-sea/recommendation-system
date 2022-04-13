import simpy
from loguru import logger
import numpy as np
import pandas as pd
import math
from cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from phase import DelayPhase, ComputePhase, IOPhase, name_app


"""TODO LIST:

            [OK] add start_delay as app parameter
            [OK] rename app.run(tiers <- placement)
            [OK] keep self.store internal
            [OK] superimpose two apps
            [OK] add id or name for each app and spread it in logs and monitoring
            
"""


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
        delay_phase = DelayPhase(duration, data=self.data, appname=self.name)
        self.store.put(delay_phase)

    def put_compute(self, duration, cores=1):
        # self.env.process(run_compute_phase(cluster, self.env, duration, cores=cores))
        # store.put(run_compute_phase(cluster, self.env, duration, cores=cores))
        compute_phase = ComputePhase(duration, cores, data=self.data, appname=self.name)
        self.store.put(compute_phase)

    def put_io(self, operation, volume, pattern=1):
        # self.env.process(run_io_phase(cluster, self.env, volume))
        # store.put(run_io_phase(cluster, self.env, volume))
        io_phase = IOPhase(operation=operation, volume=volume, pattern=pattern, data=self.data, appname=self.name)
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
                # read_io = IOPhase(operation='read', volume=self.read[i])
                # self.store.put(read_io)
                self.status.append(False)
            # then write
            if self.write[i] > 0:
                self.put_io(operation="write", volume=self.write[i])
                # write_io = IOPhase(operation='write', volume=self.write[i])
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
            if isinstance(item, DelayPhase) and phase == 0:
                self.status[phase] = yield self.env.process(item.run(self.env, cluster))
                phase += 1
            elif isinstance(item, ComputePhase):
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
                # print(f"item_number = {item_number} while tiers={tiers}")
                # print(f"status list = {self.status}")
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
# if __name__ == '__main__':
#     env = simpy.Environment()
#     data = simpy.Store(env)
#     # env.process(run_compute_phase(cluster, env, duration=10, cores=3))
#     nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
#                        'write': {'seq': 515, 'rand': 505}}
#     ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
#                      'write': {'seq': 100, 'rand': 100}}

#     ssd_tier = Tier(env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
#     nvram_tier = Tier(env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)
#     cluster = Cluster(env,  compute_nodes=1, cores_per_node=2, tiers=[ssd_tier, nvram_tier])
#     app1 = Application(env,
#                        compute=[0, 10],
#                        read=[1e9, 0],
#                        write=[0, 5e9],
#                        data=data)
#     app2 = Application(env,
#                        compute=[0],
#                        read=[3e9],
#                        write=[0],
#                        data=data)

#     # app2 = Application(env, store,
#     #                    compute=[0, 25],
#     #                    read=[2e9, 0],
#     #                    write=[0, 10e9],
#     #                    tiers=[0, 1])
#     env.process(app1.run(cluster, tiers=[0, 0]))
#     env.process(app2.run(cluster, tiers=[1, 1]))
#     env.run()
#     # print(cluster.compute_cores.capacity)
#     # print(cluster.compute_cores.data)
#     for item in data.items:
#         print(item)
    # app.put_compute(duration=10, cores=2)
    # app.put_io(volume=2e9)
    # job.put_compute(duration=10, cores=2)
    # env.process(run_io_phase(cluster, env, 10e9))
    # env.process(app.run(cluster))
    # env.run(until=20)
