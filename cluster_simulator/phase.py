#!/usr/bin/env python
"""
This module proposes a class for each type of phase that composes an application. While real applications do rarely exhibits pure phases and that are often dominant, we consider that is possible to describe it as a combinations of pure phases of compute, read or write without loss of generality.
"""


__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import simpy
from loguru import logger
import numpy as np
import pandas as pd
import math
from cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size
from cluster_simulator.cluster import EphemeralTier  # proper import for isinstance but causes import error for notebooks
#from cluster import EphemeralTier
import random
import string
import time
from simpy.events import AnyOf, AllOf, Event
from utils import monitor_step


class DelayPhase:
    """Defining an application phase of type delay which consists only on waiting a duration in order to start the next phase.

    Attributes:
        duration: duration in seconds of the waiting period.
        data: data store object where application records are kept.
        appname: a user specified application name the phase belongs to.
    """

    def __init__(self, duration, data=None, appname=None):
        """Initiates a DelayPhase with attributes."""
        self.duration = duration
        self.data = data or None
        self.appname = appname or ''

    def run(self, env, cluster):
        """Executes the delay phase by running a simple timeout event.

        Args:
            env (simpy.Environment): environment object where all simulation takes place.
            cluster (Cluster): the cluster on which the phase will run.

        Returns:
            bool: True if the execution succeeds, False if not.

        Yields:
            event: yields a timeout event.
        """
        logger.info(f"(App {self.appname}) - Start waiting phase at {env.now}")
        t_start = env.now
        yield env.timeout(self.duration)
        t_end = env.now
        monitor_step(self.data,
                     {"app": self.appname, "type": "wait", "cpu_usage": 0,
                      "t_start": t_start, "t_end": t_end, "bandwidth": 0,
                      "phase_duration": self.duration, "volume": 0,
                      "tiers": [tier.name for tier in cluster.tiers],
                      "data_placement": None,
                      "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}})
        logger.info(f"(App {self.appname}) - End waiting phase at {env.now}")
        return True


class ComputePhase:
    """Defining an application phase of type compute which consists only on doing a dominant/intensive compute operation of a certain duration involving some compute units.

    Attributes:
        duration: duration in seconds of the computing period.
        cores: number of compute units involved or consumed during the operation.
        data: data store object where application records are kept.
        appname: a user specified application name the phase belongs to.
    """

    def __init__(self, duration, cores=1, data=None, appname=None):
        """Initiates a Compute Phase instance with attributes."""
        self.duration = duration
        self.cores = cores
        self.data = self.data = data or None
        self.appname = appname or ''

    def run(self, env, cluster):
        """Executes the compute phase.

        Args:
            env (simpy.Environment): environment object where all simulation takes place.
            cluster (Cluster): the cluster on which the phase will run.

        Returns:
            bool: True if the execution succeeds, False if not.

        Yields:
            event: yields a timeout event.
        """
        logger.info(f"(App {self.appname}) - Start computing phase at {env.now} with {self.cores} requested cores")
        phase_duration = self.duration/compute_share_model(self.cores)
        t_start = env.now
        yield env.timeout(phase_duration)
        t_end = env.now
        monitor_step(self.data,
                     {"app": self.appname, "type": "compute", "cpu_usage": self.cores,
                      "t_start": t_start, "t_end": t_end, "bandwidth": 0,
                      "phase_duration": phase_duration, "volume": 0,
                      "tiers": [tier.name for tier in cluster.tiers],
                      "data_placement": None,
                      "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}})

        logger.info(f"(App {self.appname}) - End computing phase at {env.now} and releasing {self.cores} cores")
        return True


class IOPhase:
    """Defining an application I/O phase which consists on processing dominantly inputs/outputs during the application runtime.

    Attributes:
        cores: number of compute units involved or consumed during the operation.
        operation: specify if it is "read" operation (input data) or "write" operation (output data).
        volume: volume in bytes of the data to be operated during the phase.
        pattern: describes the pattern encoding with 1 if pure sequential, 0 if pure random, and a float in between.
        data: data store object where application records are kept.
        appname: a user specified application name the phase belongs to.
    """

    def __init__(self, cores=1, operation='read', volume=1e9, pattern=1, data=None, appname=None):
        """Inits an instance of I/O phase."""
        self.cores = cores
        self.operation = operation
        assert self.operation in ['read', 'write']
        self.volume = volume
        self.pattern = pattern
        self.bandwidth_usage = 1
        self.data = data or None
        self.appname = appname or ''
        # logger.info(self.__str__())

    def __str__(self):
        """Print in a human readable way a description of the I/O Phase.

        Returns:
            string: description of I/O properties.
        """
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
            yield tier.capacity.put(self.volume - tier.capacity.level)
        if self.operation == "write":
            if volume > 0:
                yield tier.capacity.put(volume)
            elif volume < 0:
                yield tier.capacity.get(abs(volume))

    def run_step(self, last_event, next_event, time_to_complete):
        if 0 < next_event - last_event < time_to_complete:
            step_duration = next_event - last_event
        else:
            step_duration = time_to_complete
        # return step_duration
        # yield self.env.timeout(step_duration)
        return step_duration

    def run(self, env, cluster, placement, use_bb=False, delay=0):
        # get the tier where the I/O will be performed
        tier = get_tier(cluster, placement, use_bb=use_bb)
        if isinstance(tier, EphemeralTier):
            # if target is ephemeral, buffer the I/O in tier
            ret = yield env.process(self.run_stage(env, cluster, tier, delay=delay))
            if ret is True:
                # if I/O is successful, destage on persistent tier
                ret2 = yield env.process(self.run_stage(env, cluster, tier.persistent_tier, delay=delay))
                return ret2
        else:
            ret = yield env.process(self.run_stage(env, cluster, tier, delay=delay))
            return ret

    def run_stage(self, env, cluster, tier, delay=0):
        # TODO : known bug when async IO reproduced in test_many_concurrent_phases_with_delay

        # get the max bandwidth available in the tier
        # TODO make a method maybe in cluster class?
        max_bandwidth = (tier.max_bandwidth[self.operation]['seq'] * self.pattern +
                         tier.max_bandwidth[self.operation]['rand'] * (1-self.pattern)) * self.cores*1e6
        # TODO, if switches are upper borne, we need to adjust the max_bandwidth
        # max_bandwidth = max(max_bandwidth, switch_bandwidth)
        # contention model : share equally available bandwidth
        volume = self.volume
        self.env = env
        #last_event = 0
        self.env.last_event = 0
        # TODO : monitor bandwidth.count?
        if delay:
            yield self.env.timeout(delay)
        # retry IO until its volume is consumed
        # next_event = last_event  # self.env.peek()
        self.env.next_event = self.env.last_event
        end_event = self.env.event()
        while volume > 0:
            with tier.bandwidth.request() as req:
                yield req
                self.bandwidth_usage = tier.bandwidth.count
                # Available bandiwidth should be f(max_bandwidth, count)
                available_bandwidth = max_bandwidth/self.bandwidth_usage
                #next_event = self.env.peek()
                self.env.next_event = min(self.env.peek(), self.env.next_event) if self.env.next_event > 0 else self.env.peek()

                # print(f"{self.appname}(start) | last_event = {self.env.last_event} | next_event = {self.env.next_event} | peek event = {self.env.peek()}")
                # take the smallest step, step_duration must be > 0
                # print(f"at {self.env.now} | last_event = {last_event} | next_event {next_event} | peek={self.env.peek()} | conc={tier.bandwidth.count}")
                #step_duration = self.run_step(last_event, next_event, volume/available_bandwidth)
                step_duration = self.run_step(self.env.last_event, self.env.next_event, volume/available_bandwidth)

                step_event = self.env.timeout(step_duration)
                self.env.next_event = min(self.env.peek(), self.env.next_event) if self.env.next_event > 0 else self.env.peek()
                # print(f"{self.appname}(mid, step_duration={step_duration}) | last_event = {self.env.last_event} | next_event = {self.env.next_event} | peek event = {self.env.peek()}")
                # print(f"{self.appname}(mid, step_duration={step_duration}) | last_event = {last_event} | next_event = {next_event} | peek event = {self.env.peek()}")
                t_start = self.env.now
                self.env.last_event = t_start
                yield step_event
                t_end = self.env.now
                #last_event = t_end
                self.env.last_event = t_end

                yield self.env.process(self.update_tier(tier, step_duration * available_bandwidth))
                volume -= step_duration * available_bandwidth
                monitoring_info = {"app": self.appname, "type": self.operation,
                                   "cpu_usage": self.cores,
                                   "t_start": t_start, "t_end": t_end,
                                   "bandwidth_concurrency": self.bandwidth_usage,
                                   "bandwidth": available_bandwidth/1e6, "phase_duration": t_end-t_start,
                                   "volume": step_duration * available_bandwidth,
                                   "tiers": [tier.name for tier in cluster.tiers],
                                   "data_placement": {"placement": tier.name},
                                   "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}}
                # print(f"{self.appname}(end, step_duration={step_duration}) | last_event = {self.env.last_event} | next_event = {self.env.next_event} | peek event = {self.env.peek()}")
                # print(f"{self.appname}(end, step_duration={step_duration}) | last_event = {last_event} | next_event = {next_event} | peek event = {self.env.peek()}")
                # # when cluster include bb tier
                if cluster.ephemeral_tier:
                    monitoring_info.update({cluster.ephemeral_tier.name + "_level": cluster.ephemeral_tier.capacity.level})
                monitor_step(self.data, monitoring_info)

                # if volume <= 0:
                #     end_event.succeed()
                #     print("Event finished")
                #     next_event = self.env.peek()
                # print(f"at {self.env.now} | last_event = {last_event} | next_event {next_event} | peek={self.env.peek()} | conc={tier.bandwidth.count} | step_duration = {self.run_step(last_event, next_event, volume/availble_bandwidth)}")
                # next_event = self.env.peek()
                # last_event += step_duration
        return True


if __name__ == '__main__':
    print("coco")
