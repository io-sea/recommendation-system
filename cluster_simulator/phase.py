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
    state = "\n | Monitoring"
    for key, value in lst.items():
        state += "| " + key + ": " + str(value) + " "
    logger.debug(state)
    if isinstance(data, simpy.Store):
        data.put(lst)
    # log_step(lst)


# def log_step(lst):

#     # {"app": self.appname, "type": "compute", "cpu_usage": self.cores,
#     #     "t_start": t_start, "t_end": t_end, "bandwidth": 0,
#     #     "phase_duration": phase_duration, "volume": 0,
#     #     "tiers": [tier.name for tier in cluster.tiers],
#     #     "data_placement": None,
#     #     "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}}
#     placement = lst["data_placement"]["placement"] if lst["data_placement"] else "None"
#     logger.info(
#         f"App {lst["app"]} | Phase: {lst["type"]} | Time: {lst["t_start"]}-->{lst["t_end"]}"
#         f"({lst["duration"]}s) | Volume = {lst["volume"]} in tier {placement}")


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
        self.bandwidth_usage = 1
        self.data = data if data else None
        self.appname = appname if appname else ''
        # logger.info(self.__str__())

    def __str__(self):
        io_pattern = f"{self.pattern*100}% sequential | {(1-self.pattern)*100} % random"
        description = "-------------------\n"
        description += (f"{self.operation.capitalize()} I/O Phase of volume {convert_size(self.volume)} with pattern: {io_pattern}\n")
        return description

    def update_tier(self, tier, volume):
        """Update tier level with the algebric value of volume.

        Args:
            tier (Tier): tier for which the level will be updated.
            volume (float): volume value (positive or negative) to adjust tier level.
        """
        # assert isinstance(tier, Tier)
        # reading operation suppose at least some volume in the tier
        if self.operation == "read" and tier.capacity.level < self.volume:
            tier.capacity.put(self.volume - tier.capacity.level)
        if self.operation == "write":
            if volume > 0:
                tier.capacity.put(volume)
            elif volume < 0:
                tier.capacity.get(abs(volume))

    def run_step(self, last_event, next_event, time_to_complete):
        if 0 < next_event - last_event < time_to_complete:
            step_duration = next_event - last_event
        else:
            step_duration = time_to_complete
        # return step_duration
        # yield self.env.timeout(step_duration)
        return step_duration

    def run(self, env, cluster, placement, delay=0):
        # TODO : known bug when async IO reproduced in test_many_concurrent_phases_with_delay
        # get the tier where the I/O will be performed
        tier = get_tier(cluster, placement)
        # get the max bandwidth available in the tier
        max_bandwidth = (tier.max_bandwidth[self.operation]['seq'] * self.pattern +
                         tier.max_bandwidth[self.operation]['rand'] * (1-self.pattern)) * self.cores*1e6
        # TODO, if switches are upper borne, we need to adjust the max_bandwidth
        # max_bandwidth = max(max_bandwidth, switch_bandwidth)
        # contention model : share equally available bandwidth
        volume = self.volume
        last_event = 0
        self.env = env

        if delay:
            yield self.env.timeout(delay)
        # retry IO until its volume is consumed
        next_event = self.env.peek()
        end_event = self.env.event()
        while volume > 0:
            with tier.bandwidth.request() as req:
                yield req
                self.bandwidth_usage = tier.bandwidth.count
                # Available bandiwidth should be f(max_bandwidth, count)
                available_bandwidth = max_bandwidth/self.bandwidth_usage

                next_event = self.env.peek()

                # take the smallest step, step_duration must be > 0
                # print(f"at {self.env.now} | last_event = {last_event} | next_event {next_event} | peek={self.env.peek()} | conc={tier.bandwidth.count}")
                step_duration = self.run_step(last_event, next_event, volume/available_bandwidth)
                step_event = self.env.timeout(step_duration)
                t_start = self.env.now
                yield step_event
                t_end = self.env.now
                self.update_tier(tier, step_duration * available_bandwidth)
                volume -= step_duration * available_bandwidth
                # TODO update tier state
                monitor(self.data,
                        {"app": self.appname, "type": self.operation, "cpu_usage": self.cores,
                         "t_start": t_start, "t_end": t_end,
                         "bandwidth_concurrency": self.bandwidth_usage,
                         "bandwidth": available_bandwidth/1e6, "phase_duration": t_end-t_start,
                         "volume": convert_size(step_duration * available_bandwidth),
                         "tiers": [tier.name for tier in cluster.tiers],
                         "data_placement": {"placement": tier.name},
                         "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}})

                # if volume <= 0:
                #     end_event.succeed()
                #     print("Event finished")
                #     next_event = self.env.peek()
                # print(f"at {self.env.now} | last_event = {last_event} | next_event {next_event} | peek={self.env.peek()} | conc={tier.bandwidth.count} | step_duration = {self.run_step(last_event, next_event, volume/available_bandwidth)}")
                # next_event = self.env.peek()
                # last_event += step_duration

        return True
