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
#from cluster_simulator.cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size

from cluster_simulator.utils import convert_size, get_tier, compute_share_model
from cluster_simulator.cluster import Tier
from cluster_simulator.cluster import EphemeralTier  # proper import for isinstance but causes import error for notebooks
#from cluster import EphemeralTier
import random
import string
import time
from simpy.events import AnyOf, AllOf, Event


def monitor_step(data, lst):
    """Monitoring function that feed a queue of records on phases events when an application is running on the cluster.

    Args:
        data (simpy.Store): a store object that queues elements of information useful for logging and analytics.
        lst (dict): information element to add to the data store.
    """
    state = "\n | Monitoring"
    for key, value in lst.items():
        state += f"| {key}: {str(value)} "
    logger.debug(state)
    if isinstance(data, simpy.Store):
        data.put(lst)


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
    Variables:
        current_ios: list of IOPhase instances that are in running state so it is possible to update IOs following a change in bandwidth consumption.
    Attributes:
        cores: number of compute units involved or consumed during the operation.
        operation: specify if it is "read" operation (input data) or "write" operation (output data).
        volume: volume in bytes of the data to be operated during the phase.
        pattern: describes the pattern encoding with 1 if pure sequential, 0 if pure random, and a float in between.
        last_event: float to keep the timestamp of the last I/O event.
        next_event: float to keep the timestamp of the next I/O event.
        data: data store object where application records are kept.
        appname: a user specified application name the phase belongs to.
    """
    #
    current_ios = []

    def __init__(self, cores=1, operation='read', volume=1e9, pattern=1, data=None, appname=None):
        """Inits an instance of I/O phase."""
        self.cores = cores
        self.operation = operation
        assert self.operation in ['read', 'write']
        self.volume = volume
        self.pattern = pattern
        self.last_event = 0
        self.next_event = 0
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

    def register_step(self, t_start, step_duration, available_bandwidth, cluster, tier):
        """Registering a processing step in the data store with some logging.

        Args:
            t_start (float): timestamp of the start of the step.
            step_duration (float): duration of the step.
            available_bandwidth (float): available bandwidth in the step.
            cluster (Cluster): the cluster on which the phase will run.
            tier (Tier): the tier on which the step will run.
        """
        monitoring_info = {"app": self.appname, "type": self.operation,
                           "cpu_usage": self.cores,
                           "t_start": t_start, "t_end": t_start + step_duration,
                           "bandwidth_concurrency": tier.bandwidth.count,
                           "bandwidth": available_bandwidth/1e6,
                           "phase_duration": step_duration,
                           "volume": step_duration * available_bandwidth,
                           "tiers": [tier.name for tier in cluster.tiers],
                           "data_placement": {"placement": tier.name},
                           "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}}
        # update info if using a burst buffer
        if cluster.ephemeral_tier:
            monitoring_info[cluster.ephemeral_tier.name + "_level"] = cluster.ephemeral_tier.capacity.level
        monitor_step(self.data, monitoring_info)

    def process_volume(self, step_duration, volume, available_bandwidth, cluster, tier):
        """This method processes a small amount of I/O volume between two predictable events. If an event occurs in the meantime, I/O will be interrupted and bandwidth updated according.

        Args:
            step_duration (float): the expected duration between two predictible events.
            volume (float): volume in bytes of the data to process.
            cluster (Cluster): cluster facility where the I/O operation should take place.
            available_bandwidth (float): available bandwidth in the step.
            tier (Tier): storage tier concerned by the I/O operation. It could be reading from this tier or writing to it.
        """
        try:
            t_start = self.env.now
            yield self.env.timeout(step_duration)
            self.last_event += step_duration
            volume -= step_duration * available_bandwidth
            self.update_tier(tier, step_duration * available_bandwidth)
            self.register_step(t_start, step_duration, available_bandwidth, cluster, tier)

        except simpy.exceptions.Interrupt as interrupt:
            logger.info('interrupt')
            t_end = self.env.now
            step_duration = t_end - start
            volume -= step_duration * available_bandwidth
            self.update_tier(tier, step_duration * available_bandwidth)
            self.register_step(t_start, step_duration, available_bandwidth, cluster, tier)

        return volume

    def update_tier(self, tier, volume):
        """Update tier level with the algebric value of volume.

        Args:
            tier (Tier): tier for which the level will be updated.
            volume (float): volume value (positive or negative) to adjust tier level.
        """
        assert isinstance(tier, Tier)
        # reading operation suppose at least some volume in the tier
        if self.operation == "read" and tier.capacity.level < self.volume:
            tier.capacity.put(self.volume - tier.capacity.level)
        if self.operation == "write" and volume > 0:
            tier.capacity.put(volume)
        elif volume < 0:
            tier.capacity.get(abs(volume))

    def run_step(self, env, cluster, tier):
        """Allows to run a step of I/O operation where bandwidth share is constant.

        Args:
            env (simpy.Environment()): environment variable where the I/O operation will take place.
            cluster (Cluster): the cluster on which the application will run.
            tier (Tier): the storage tier where the I/O will be processed.

        Returns:
            bool: True if the step is completed, False otherwise.

        Yields:
            simpy.Event: other events that can occur during the I/O operation.
        """
        volume = self.volume
        self.env = env
        # retry IO until its volume is consumed
        while volume > 0:
            with tier.bandwidth.request() as req:
                yield req
                max_bandwidth = cluster.get_max_bandwidth(tier, operation=self.operation, pattern=self.pattern)
                available_bandwidth = max_bandwidth/tier.bandwidth.count
                self.next_event = self.env.peek()
                # take the smallest step, step_duration must be > 0
                if 0 < self.next_event - self.last_event < volume/available_bandwidth:
                    step_duration = self.next_event - self.last_event
                else:
                    step_duration = volume/available_bandwidth
                step_event = self.env.process(self.process_volume(step_duration, volume, available_bandwidth, cluster, tier))
                # register the step event to be able to update it
                IOPhase.current_ios.append(step_event)
                # process the step volume
                volume = yield step_event

        return True

    def run(self, env, cluster, placement, use_bb=False, delay=0):
        self.env = env
        if delay:
            yield self.env.timeout(delay)
        # get the tier where the I/O will be performed
        tier = get_tier(cluster, placement, use_bb=use_bb)
        if isinstance(tier, EphemeralTier):
            # if target is ephemeral, buffer the I/O in tier
            ret = yield self.env.process(self.run_step(self.env, cluster, tier))
            if ret is True:
                # if I/O is successful, destage on persistent tier
                ret2 = yield self.env.process(self.run_step(self.env, cluster, tier.persistent_tier))
                return ret2
        else:
            ret = yield env.process(self.run_step(env, cluster, tier))
            return ret


class BandwidthResource(simpy.Resource):
    """Subclassing simpy Resource to introduce the ability to check_bandwidth when resource is requested or released."""

    def __init__(self, *args, **kwargs):
        """Init method using parent init method."""
        super().__init__(*args, **kwargs)
        self.env = args[0]

    def request(self, *args, **kwargs):
        """On request method, cehck_bandwidth using parent request method."""
        self.check_bandwidth()
        return super().request(*args, **kwargs)

    def release(self, *args, **kwargs):
        """On release method, check_bandwidth using parent release method."""
        self.check_bandwidth()
        return super().release(*args, **kwargs)

    def check_bandwidth(self):
        """Checks running IO when bandwidth occupation changes. IOs should be interrupted on release or request of a bandwidth slot.
        """
        for io_event in IOPhase.current_ios:
            if not io_event.processed and io_event.triggered and io_event.is_alive:
                # capture the IOs not finished, but triggered and alive
                print(io_event.is_alive)
                print(io_event.value)
                io_event.interrupt('updating bandwidth')


if __name__ == '__main__':
    print("coco")
