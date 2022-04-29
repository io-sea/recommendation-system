import simpy
from loguru import logger
import numpy as np
import pandas as pd
import math
from cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
import random
import string
import time
from simpy.events import AnyOf, AllOf, Event


def monitor(data, lst):
    state = "--Monitoring"
    for key, value in lst.items():
        state += "| " + key + ": " + str(value) + " "
    logger.debug(state)
    if isinstance(data, simpy.Store):
        data.put(lst)


def name_app():
    return ''.join(random.sample(string.ascii_uppercase, 1)) + str(random.randint(0, 9))


class DelayPhase:
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


class ComputePhase:
    def __init__(self, duration, cores=1, data=None, appname=None):
        self.duration = duration
        self.cores = cores
        self.data = data if data else None
        self.appname = appname if appname else ''

    def run(self, env, cluster):
        used_cores = []
        # use self.cores
        # for i in range(self.cores):
        #     core = cluster.compute_cores.request()
        #     used_cores.append(core)
        #     yield core
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

        # releasing cores
        # for core in used_cores:
        #     cluster.compute_cores.release(core)

        logger.info(f"(App {self.appname}) - End computing phase at {env.now} and releasing {self.cores} cores")
        return True


class IOPhase:
    def __init__(self, cores=1, operation='read', volume=1e9, pattern=1, data=None, appname=None):
        """
        pattern = 0.8:
            80% sequential and 20% random.
            blocksize for moment is not variable."""
        self.cores = cores
        self.operation = operation
        assert self.operation in ['read', 'write']
        self.volume = volume
        self.pattern = pattern
        self.bandwidth_usage = dict()
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
        # assert isinstance(tier, Tier)
        # reading operation suppose at least some volume in the tier
        if self.operation == "read" and tier.capacity.level < self.volume:
            tier.capacity.put(self.volume - tier.capacity.level)
        if self.operation == "write":
            tier.capacity.put(self.volume)

    def log_phase_start(self, timestamp=0, io_bandwidth=None, max_bandwidth=None):

        logger.info(f"(App {self.appname}) - Start {self.operation.capitalize()} I/O Phase with volume = {convert_size(self.volume)} at {timestamp}")

        # logger.info(f"(App {self.appname}) - {self.operation.capitalize()}(ing) I/O with bandwidth = {max_bandwidth} MB/s available at {timestamp}")
        logger.info(f"(App {self.appname}) - {self.operation.capitalize()}(ing) available bandwidth = {round(io_bandwidth, 2)} MB/s available at {timestamp}")

        logger.info(f"(App {self.appname}) - {self.operation.capitalize()}(ing) I/O with bandwidth = {io_bandwidth} MB/s")

    def log_phase_end(self, timestamp):
        logger.info(f"(App {self.appname}) - End {self.operation.capitalize()} I/O Phase at {timestamp}")

    # def run(self, env, cluster, placement=None):
    #     """Run the I/O phase within a cluster and according to a specifier placement in available tiers."""
    #     tier = get_tier(placement, cluster)  # get the tier from placement list
    #     max_tier_bandwidth = (tier.max_bandwidth[self.operation]['seq'] * self.pattern +
    #                           tier.max_bandwidth[self.operation]['rand']*(1-self.pattern)) * self.cores
    #     # requesting bandwidth, avoiding contention model : take all available
    #     remaining_bandwidth = tier.bandwidth.capacity - tier.bandwidth.count
    #     if remaining_bandwidth:
    #         # booking bandwidth
    #         booking_bandwidth = [tier.bandwidth.request() for i in range(remaining_bandwidth)]
    #         # booking cores
    #         booking_cores = [cluster.compute_cores.request() for i in range(self.cores)]
    #         # adjust io bandwidth
    #         io_bandwidth = max_tier_bandwidth*remaining_bandwidth/tier.bandwidth.capacity
    #         phase_duration = (self.volume/1e6)/io_bandwidth  # bandwidth in MB/s
    #         t_start = env.now
    #         self.update_tiers(cluster, tier)
    #         booking_phase = env.timeout(phase_duration, value=True)

    def run(self, env, cluster, placement=None, delay=0):
        # get the tier where the I/O will be performed
        tier = get_tier(placement, cluster)
        # get the max bandwidth available in the tier
        max_bandwidth = (tier.max_bandwidth[self.operation]['seq'] * self.pattern +
                         tier.max_bandwidth[self.operation]['rand'] * (1-self.pattern)) * self.cores*1e6
        # TODO, if switches are upper borne, we need to adjust the max_bandwidth
        # max_bandwidth = max(max_bandwidth, switch_bandwidth)
        # contention model : share equally available bandwidth
        volume = self.volume
        self.env = env
        if delay:
            yield self.env.timeout(delay)
        time_step = 1
        # retry IO until its volume is consumed
        while volume > 0:
            with tier.bandwidth.request(priority=-time.time(), preempt=True) as req:
                yield req
                try:  # try exhausting IO volume
                    available_bandwidth = max_bandwidth/tier.bandwidth.count
                    t_start = self.env.now
                    # yield self.env.timeout(time_step)
                    yield self.env.timeout(volume/available_bandwidth)
                    t_end = self.env.now
                    self.bandwidth_usage[t_start] = available_bandwidth/1e6

                    self.log_phase_start(timestamp=env.now,
                                         io_bandwidth=available_bandwidth/1e6,           max_bandwidth=max_bandwidth)
                    available_bandwidth = max_bandwidth/tier.bandwidth.count
                    # volume -= time_step*available_bandwidth  # until 0 exit the loop

                    monitor(self.data,
                            {"app": self.appname, "type": self.operation, "cpu_usage": self.cores,
                             "t_start": t_start, "t_end": t_end, "bandwidth": available_bandwidth/1e6,
                             "phase_duration": t_end-t_start, "volume": volume,
                             "tiers": [tier.name for tier in cluster.tiers],
                             "data_placement": {"placement": tier.name},
                             "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}})
                    volume = 0
                except simpy.Interrupt as interrupt:
                    t_end = self.env.now
                    time_usage = t_end - interrupt.cause.usage_since
                    logger.info(f"duration before interruption {time_usage}")
                    available_bandwidth = max_bandwidth/tier.bandwidth.count
                    volume -= time_usage*available_bandwidth
                    monitor(self.data,
                            {"app": self.appname, "type": self.operation, "cpu_usage": self.cores,
                             "t_start": interrupt.cause.usage_since, "t_end": t_end, "bandwidth": available_bandwidth/1e6,
                             "phase_duration": t_end-t_start, "volume": time_usage*available_bandwidth,
                             "tiers": [tier.name for tier in cluster.tiers],
                             "data_placement": {"placement": tier.name},
                             "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}})
        self.log_phase_end(timestamp=t_end)
        return True
