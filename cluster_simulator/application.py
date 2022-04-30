import simpy
from loguru import logger
import numpy as np
import pandas as pd
import math
from cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from phase import DelayPhase, ComputePhase, IOPhase, name_app
import copy
from simpy.events import AnyOf, AllOf, Event
from loguru import logger
import math

"""TODO LIST:

            [OK] add start_delay as app parameter
            [OK] rename app.run(tiers <- placement)
            [OK] keep self.store internal
            [OK] superimpose two apps
            [OK] add id or name for each app and spread it in logs and monitoring
            [  ] bandwidth is a preemptible resource

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
        self.cores_request = []
        self.schedule()

    def put_delay(self, duration):
        delay_phase = DelayPhase(duration, data=self.data, appname=self.name)
        self.store.put(delay_phase)
        self.cores_request.append(0)

    def put_compute(self, duration, cores=1):
        # self.env.process(run_compute_phase(cluster, self.env, duration, cores=cores))
        # store.put(run_compute_phase(cluster, self.env, duration, cores=cores))
        compute_phase = ComputePhase(duration, cores, data=self.data, appname=self.name)
        self.store.put(compute_phase)
        self.cores_request.append(compute_phase.cores)

    def put_io(self, operation, volume, pattern=1):
        # self.env.process(run_io_phase(cluster, self.env, volume))
        # store.put(run_io_phase(cluster, self.env, volume))
        io_phase = IOPhase(cores=1, operation=operation, volume=volume, pattern=pattern, data=self.data, appname=self.name)
        self.store.put(io_phase)
        self.cores_request.append(io_phase.cores)

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

    def request_cores(self, cluster):

        return [cluster.compute_cores.request() for i in range(max(self.cores_request))]

    def run(self, cluster, tiers):
        # assert len(cluster.tiers) == len(tiers)
        item_number = 0
        phase = 0
        requesting_cores = self.request_cores(cluster)
        while self.store.items:
            item = yield self.store.get()
            if isinstance(item, DelayPhase) and phase == 0:
                self.status[phase] = yield self.env.process(item.run(self.env, cluster))
                phase += 1
            elif isinstance(item, ComputePhase):
                # compute phase
                if phase == 0:
                    yield AllOf(self.env, requesting_cores)
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster))
                    phase += 1
                elif phase > 0 and self.status[phase-1] == True:
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster))
                    phase += 1
                else:
                    self.status[phase] = False
            else:
                placement = cluster.tiers[tiers[item_number]]
                if phase == 0:
                    yield AllOf(self.env, requesting_cores)
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster, placement=placement))
                    phase += 1
                elif phase > 0 and self.status[phase-1] == True:
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster, placement=placement))
                    phase += 1
                else:
                    self.status[phase] = False
                    phase += 1
                item_number += 1

        releasing_cores = [cluster.compute_cores.release(core) for core in requesting_cores]
        yield AllOf(self.env, releasing_cores)
        return self.data

    def get_fitness(self, app_name_filter=None):
        """Method to get app duration from execution records saved in data store"""
        # sample_item = {'app': 'B8', 'type': 'read', 'cpu_usage': 1, 't_start': 0, 't_end': 4.761904761904762, 'bandwidth': 210.0, 'phase_duration': 4.761904761904762, 'volume': 1000000000.0, 'tiers': ['SSD', 'NVRAM'], 'data_placement': {'placement': 'SSD'}, 'tier_level': {'SSD': 1000000000.0, 'NVRAM': 0}}
        t_max = 0
        if not self.data:
            logger.error("No data store provided")
            return None
        if not self.data.items:
            return t_max
        for phase in self.data.items:
            if app_name_filter is not None:
                if phase["app"] == app_name_filter:
                    t_max = max(t_max, phase["t_end"])
            else:
                t_max = max(t_max, phase["t_end"])
        return t_max


class IO:
    def __init__(self, env, name, volume, bandwidth, delay, prio):
        self.env = env
        self.name = name
        self.volume = volume
        self.bandwidth = bandwidth
        self.delay = delay
        self.prio = prio
        self.concurrent = False
        self.b_usage = dict()
        self.last_event = 0
        self.process = env.process(self.run())

    def run(self):
        yield self.env.timeout(self.delay)
        # remaining volume of the IO to be conveyed
        volume = self.volume
        # retry IO until its volume is consumed
        while volume > 0:

            with self.bandwidth.request() as req:

                yield req
                self.b_usage[self.env.now] = round(100/self.bandwidth.count, 2)
                # try exhausting IO volume
                # update bandwidth usage
                available_bandwidth = 1/(self.bandwidth.count+len(self.bandwidth.queue))
                start = self.env.now
                step_duration = min(self.env.peek() - self.last_event, volume/available_bandwidth)
                logger.info(f"env.peek = {self.env.peek()}")
                yield self.env.timeout(step_duration)
                self.last_event += step_duration
                volume -= step_duration * available_bandwidth
                #available_bandwidth = 1/self.bandwidth.count
                logger.info(f"[{self.name}](step) time "
                            f"= {start}-->{start+step_duration} | "
                            f"remaining volume = {volume} | "
                            f"available_bandwidth : {available_bandwidth} ")
                self.b_usage[env.now] = round(100/self.bandwidth.count, 2)
            #self.b_usage[env.now] = round(100/self.bandwidth.count, 2)

        # except simpy.Interrupt as interrupt:
        #     self.concurrent = True
        #     logger.info(f"[{self.name}](preempted) at {self.env.now} | "
        #                 f"by {interrupt.cause.by}| available_bandwidth = {available_bandwidth}")
        #     # update volume
        #     time_usage = self.env.now - interrupt.cause.usage_since
        #     volume -= time_usage*available_bandwidth
        #     # update bandiwdth
        #     available_bandwidth = 1/self.bandwidth.count
        #     self.prio -= 1
        #     logger.info(f"[{self.name}](consuming) {time_usage*available_bandwidth} out of {self.volume} | current volume = {volume}")
        #     self.b_usage[env.now] = round(100/bandwidth.count, 2)
        # logger.info(f"[{self.name}](consumed) start = {start} end = {self.env.now} | "
        #             f"volume = {self.volume}-->{volume} |")
        # print(self.b_usage)


def resource_user(name, env, resource, wait, prio):
    b_usage = dict()
    yield env.timeout(wait)
    volume = 2
    while volume > 0:
        req = resource.request(priority=prio)
        print('%s requesting at %s with priority=%s | count=%s' % (name, env.now, prio, resource.count))
        b_usage[env.now] = resource.count
        yield req
        print('%s got resource at %s | count=%s | bandwidth=%s%%' % (name, env.now, resource.count, round(100/resource.count, 1)))
        b_usage[env.now] = resource.count
        try:
            print(f"next event will take at {env.peek()}")
            if volume <= env.peek():  # volume consumed before event
                yield env.timeout(volume)
                volume = 0
            else:
                yield env.timeout(env.peek())
                volume -= env.peek()

            b_usage[env.now] = resource.count
        except simpy.Interrupt as interrupt:
            by = interrupt.cause.by
            usage = env.now - interrupt.cause.usage_since
            volume -= usage
            prio -= 0.1  # bump my prio enough so I'm next
            print('%s got preempted by %s at %s after %s | count=%s' %
                  (name, by, env.now, usage, resource.count))
            b_usage[env.now] = resource.count
    print('%s completed at time %g | count=%s | bandwidth=%s%%' % (name, env.now, resource.count, round(100/resource.count, 1)))
    print(b_usage)


def io_run(name, env, resource, wait, prio):
    b_usage = dict()
    yield env.timeout(wait)
    volume = 2
    while volume > 0:
        with resource.request(priority=prio) as req:
            print('%s requesting at %s with priority=%s | count=%s' % (name, env.now, prio, resource.count))
            b_usage[env.now] = resource.count
            yield req
            print('%s got resource at %s | count=%s | bandwidth=%s%%' % (name, env.now, resource.count, round(100/resource.count, 1)))
            b_usage[env.now] = resource.count
            try:
                print(f"next event will take at {env.peek()}")
                if volume/resource.count <= env.peek():  # volume consumed before event
                    yield env.timeout(volume*resource.count)
                    volume = 0
                    print('%s completed at time %g | count=%s | bandwidth=%s%%' % (name, env.now, resource.count, round(100/resource.count, 1)))
                else:
                    yield env.timeout(env.peek())
                    volume -= env.peek()/resource.count
                    if volume <= 0:
                        print('%s completed at time %g | count=%s | bandwidth=%s%%' % (name, env.now, resource.count, round(100/resource.count, 1)))

                b_usage[env.now] = resource.count
            except simpy.Interrupt as interrupt:
                by = interrupt.cause.by
                usage = env.now - interrupt.cause.usage_since
                volume -= usage
                # prio -= 0.1  # bump my prio enough so I'm next
                print('%s got preempted by %s at %s after %s | count=%s' %
                      (name, by, env.now, usage, resource.count))
                b_usage[env.now] = resource.count

    print(b_usage)


if __name__ == '__main__':
    # env = simpy.Environment()
    # res = simpy.PreemptiveResource(env, capacity=10)
    # p1 = env.process(io_run("app1", env, res, wait=0, prio=0))
    # p2 = env.process(io_run("app2", env, res, wait=1, prio=0))
    # #p3 = env.process(resource_user("app3", env, res, wait=0, prio=-1))
    # env.run()
    env = simpy.Environment()
    bandwidth = simpy.PreemptiveResource(env, capacity=10)
    #bandwidth = simpy.PreemptiveResource(env, capacity=2)
    IOs = [IO(env, name=str(i), volume=2,
              bandwidth=bandwidth, delay=i*1, prio=i) for i in range(2)]

    env.run()
    # print(bandwidth.data)
    for io in IOs:
        print(f"app: {io.name} | bandwidth usage: {io.b_usage}")

    # data = simpy.Store(env)

    # nvram_bandwidth = {'read':  {'seq': 780, 'rand': 760},
    #                    'write': {'seq': 515, 'rand': 505}}
    # ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
    #                  'write': {'seq': 100, 'rand': 100}}

    # ssd_tier = Tier(env, 'SSD', bandwidth=ssd_bandwidth, capacity=200e9)
    # nvram_tier = Tier(env, 'NVRAM', bandwidth=nvram_bandwidth, capacity=80e9)
    # cluster = Cluster(env,  compute_nodes=1, cores_per_node=2, tiers=[ssd_tier, nvram_tier])
    # app1 = Application(env,
    #                    compute=[0, 10],
    #                    read=[0, 0],
    #                    write=[0, 0],
    #                    data=data)
    # app2 = Application(env,
    #                    name="popo",
    #                    compute=[0, 15],
    #                    read=[0, 0],
    #                    write=[0, 0],
    #                    data=data)

    # # app2 = Application(env, store,
    # #                    compute=[0, 25],
    # #                    read=[2e9, 0],
    # #                    write=[0, 10e9],
    # #                    tiers=[0, 1])
    # env.process(app1.run(cluster, tiers=[0, 0]))
    # env.process(app2.run(cluster, tiers=[1, 1]))
    # env.run()

    # for item in data.items:
    #     print(item)
    #     print(get_app_duration(data, app="popo"))
    #     fig = analytics.display_run(data, cluster, width=700, height=900)
    #     fig.show()

    # item = app1.phases.get()
    # print("---")
    # print(type(item))
    # print(cluster.compute_cores.capacity)
    # print(cluster.compute_cores.data)
    # for item in data.items:
    #     print(item)
    # app.put_compute(duration=10, cores=2)
    # app.put_io(volume=2e9)
    # job.put_compute(duration=10, cores=2)
    # env.process(run_io_phase(cluster, env, 10e9))
    # env.process(app.run(cluster))
    # env.run(until=20)
