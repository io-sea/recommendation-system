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
from enum import Enum
# from cluster_simulator.cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier, convert_size

from cluster_simulator.utils import convert_size, get_tier, compute_share_model, BandwidthResource
from cluster_simulator.cluster import Tier, EphemeralTier, get_tier
from cluster_simulator.phase_features import PhaseFeatures
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
    callable_dict = {"bandwidth": lambda x: f"{str(x)} MB/s", "volume": lambda x: convert_size(x)}
    state = "\n | Monitoring"
    for key, value in lst.items():
        if key in callable_dict:
            state += f"| {key}: {callable_dict[key](value)} "
        else:
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
            env (simpy.Environment): environment object where all simulation
            takes place.
            cluster (Cluster): the cluster on which the phase will run.

        Returns:
            bool: True if the execution succeeds, False if not.

        Yields:
            event: yields a timeout event.
        """
        logger.info(f"(App {self.appname}) - Start waiting phase at {env.now}")
        t_start = env.now
        initial_levels = cluster.get_levels()
        yield env.timeout(self.duration)
        t_end = env.now
        final_levels = cluster.get_levels()
        monitor_step(self.data,
                     {"app": self.appname, "type": "wait", "cpu_usage": 0,
                      "t_start": t_start, "t_end": t_end, "bandwidth": 0,
                      "phase_duration": self.duration, "volume": 0,
                      "tiers": [tier.name for tier in cluster.tiers],
                      "data_placement": None,
                      "init_level": initial_levels,
                      "tier_level": final_levels})
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
        initial_levels = cluster.get_levels()
        yield env.timeout(phase_duration)
        t_end = env.now
        final_levels = cluster.get_levels()
        monitor_step(self.data,
                     {"app": self.appname, "type": "compute", "cpu_usage": self.cores,
                      "t_start": t_start, "t_end": t_end, "bandwidth": 0,
                      "phase_duration": phase_duration, "volume": 0,
                      "tiers": [tier.name for tier in cluster.tiers],
                      "data_placement": None,
                      "init_level": initial_levels,
                      "tier_level": final_levels})

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
        bandwidth_concurrency: int to indicate how many processes are doing I/O.
        dirty: int to indicate the amount of data that is dirty in ephemeral tier/RAM (does not have a copy in persistent tier).
        data: data store object where application records are kept.
        appname: a user specified application name the phase belongs to.
        bw : in MB/s the observed throughput for this IO to reproduce the observed results.
    """
    # list of IOPhase instances that are in running state so it is possible to update IOs following a change in bandwidth consumption
    current_ios = []

    def __init__(self, cores=1, operation='read', volume=1e9, pattern=1,
                 data=None, appname=None, bw=None, phase_features=None):
        """Inits an instance of I/O phase."""
        self.cores = cores
        self.operation = operation
        assert self.operation in ['read', 'write']
        self.volume = volume
        self.pattern = pattern
        self.last_event = 0
        self.next_event = 0
        self.bandwidth_concurrency = 1
        self.data = data or None
        self.appname = appname or ''
        self.phase_features = phase_features or None
        # Bytes to MB conversion
        # case where bandwidth is given for a reproducing a simulation
        self.bw = bw*1e6 if bw else bw
        # logger.info(self.__str__())

    def __str__(self):
        """Print in a human readable way a description of the I/O Phase.

        Returns:
            string: description of I/O properties.
        """
        io_pattern = f"{self.pattern*100}% sequential | {(1-self.pattern)*100} % random"
        description = "-------------------\n"
        description += (f"{self.operation.capitalize()} I/O Phase of volume {convert_size(self.volume)} with pattern: {io_pattern}\n")
        logger.trace(description)
        return ""

    def register_step(self, t_start, step_duration, available_bandwidth, cluster, tier,
                      initial_levels=None, source_tier=None, eviction=None):
        """Registering a processing step in the data store with some logging.

        Args:
            t_start (float): timestamp of the start of the step.
            step_duration (float): duration of the step.
            available_bandwidth (float): available bandwidth in the step.
            cluster (Cluster): the cluster on which the phase will run.
            tier (Tier): the tier on which the step will run.
            initial_levels (dict): initial levels of all tiers at the start of the step.
            source_tier (Tier, optional): the tier from which the step will run.
            eviction (int, optional): volume of data which was evicted from ephemeral tier.
        """
        final_levels = cluster.get_levels()
        monitoring_info = {"app": self.appname, "type": self.operation,
                           "cpu_usage": self.cores,
                           "t_start": t_start, "t_end": t_start + step_duration,
                           "bandwidth_concurrency": self.bandwidth_concurrency,
                           "bandwidth": available_bandwidth/1e6,
                           "phase_duration": step_duration,
                           "volume": step_duration * available_bandwidth,
                           "tiers": [tier.name for tier in cluster.tiers],
                           "data_placement": {"placement": tier.name},
                           "init_level": initial_levels,
                           "tier_level": final_levels,
                           # "tier_level": {tier.name: tier.capacity.level for tier in cluster.tiers}
                           }
        # update info if using a burst buffer
        if cluster.ephemeral_tier:
            monitoring_info[cluster.ephemeral_tier.name + "_level"] = cluster.ephemeral_tier.capacity.level
        # update info if using source_tier for data movement
        if source_tier:
            monitoring_info["data_placement"]["source"] = source_tier.name
            monitoring_info["type"] = "movement"
        if eviction:
            monitoring_info["type"] = "eviction"
            monitoring_info["t_start"] = self.env.now
            monitoring_info["t_end"] = self.env.now
            monitoring_info["bandwidth"] = float("inf")
            monitoring_info["phase_duration"] = 0
            monitoring_info["volume"] = eviction
            if source_tier:
                logger.debug(f"tier dirty: {convert_size(source_tier.dirty)} | "
                             f"tier level: {convert_size(source_tier.capacity.level)} | "
                             f"eviction volume = {convert_size(eviction)}")
            else:
                logger.debug(f"tier dirty: {convert_size(tier.dirty)} | "
                             f"tier level: {convert_size(tier.capacity.level)} | "
                             f"eviction volume = {convert_size(eviction)}")

        monitor_step(self.data, monitoring_info)

    def update_tier(self, tier, volume):
        """Update tier level with the algebric value of volume.

        Args:
            tier (Tier): tier for which the level will be updated.
            volume (float): volume value (positive or negative) to adjust tier level.
        """
        assert isinstance(tier, Tier)  # check if tier is a Tier/EphemeralTier instance
        # reading operation suppose at least some volume in the tier
        if self.operation == "read" and tier.capacity.level < self.volume:
            tier.capacity.put(self.volume - tier.capacity.level)
        if self.operation == "write" and volume > 0:
            tier.capacity.put(volume)
            if isinstance(tier, EphemeralTier):
                tier.dirty += volume
        elif volume < 0:
            tier.capacity.get(abs(volume))

    def update_tier_on_move(self, source_tier, target_tier, volume, erase):
        """Update tier level following a volume move.

        Args:
            source_tier (Tier): tier from which the data will be moved.
            target_tier (Tier): tier for which the level will be updated.
            volume (float): volume value (positive or negative) to adjust tier level.
            erase (bool): whether or not erase the amount of volume from source_tier.
        """
        assert isinstance(source_tier, Tier)
        assert isinstance(target_tier, Tier)
        # reading operation suppose at least some volume in the tier
        if source_tier.capacity.level < volume:
            source_tier.capacity.put(volume - source_tier.capacity.level)
        # target_tier get its volume updated
        #if volume > 0:
        target_tier.capacity.put(volume)
        # destaging decreases the amount of dirty data as it get a copy in a persistent tier
        if isinstance(source_tier, EphemeralTier) and type(target_tier) == Tier:
            source_tier.dirty -= volume
            source_tier.dirty = max(0, source_tier.dirty)
        # if erase original data, then update source_tier
        if erase:
            source_tier.capacity.get(volume)

    def process_volume(self, step_duration, volume, available_bandwidth, cluster, tier,
                       initial_levels=None):
        """This method processes a small amount of I/O volume between two predictable events on a specific tier. If an event occurs in the meantime, I/O will be interrupted and bandwidth updated according.

        Args:
            step_duration (float): the expected duration between two predictable events.
            volume (float): volume in bytes of the data to process.
            cluster (Cluster): cluster facility where the I/O operation should take place.
            available_bandwidth (float): available bandwidth in the step.
            tier (Tier): storage tier concerned by the I/O operation. It could be reading from this tier or writing to it.
        """
        try:
            t_start = self.env.now
            volume_event = self.env.timeout(step_duration)
            yield volume_event
            volume -= step_duration * available_bandwidth
            if not initial_levels:
                initial_levels = cluster.get_levels()
            self.update_tier(tier, step_duration * available_bandwidth)
            self.register_step(t_start, step_duration, available_bandwidth, cluster, tier=tier,
                               initial_levels=initial_levels)
            if isinstance(tier, EphemeralTier):
                initial_levels = cluster.get_levels()
                eviction = tier.evict()
                if eviction:
                    self.register_step(t_start, step_duration, available_bandwidth, cluster,
                                       tier=tier, initial_levels=initial_levels, eviction=eviction)

        except simpy.exceptions.Interrupt:
            logger.trace(f'{self.appname} interrupt at {self.env.now}')
            step_duration = self.env.now - t_start
            if step_duration:
                self.last_event += step_duration
                volume -= step_duration * available_bandwidth
                if not initial_levels:
                    initial_levels = cluster.get_levels()
                self.update_tier(tier, step_duration * available_bandwidth)
                self.register_step(t_start, step_duration, available_bandwidth, cluster, tier=tier,
                                   initial_levels=initial_levels)
                if isinstance(tier, EphemeralTier):
                    initial_levels = cluster.get_levels()
                    eviction = tier.evict()
                    if eviction:
                        self.register_step(t_start, step_duration, available_bandwidth,
                                           cluster, tier=tier, initial_levels=initial_levels, eviction=eviction)
        return volume

    def move_volume(self, step_duration, volume, available_bandwidth, cluster, source_tier,
                    target_tier, erase=False, initial_levels=None):
        """This method moves a small amount of I/O volume between two predictable events from a source_tier to a target tier with available bandiwdth value. If an event occurs in the meantime, data movement will be interrupted and bandwidth updated accordingly.

        Args:
            step_duration (float): the expected duration between two predictible events.
            volume (float): volume in bytes of the data to move.
            cluster (Cluster): cluster facility where the I/O operation should take place.
            available_bandwidth (float): available bandwidth in the step.
            source_tier (Tier): storage tier from which we read data to move.
            target_tier (Tier): storage tier where the data will be moved.
        """
        try:
            t_start = self.env.now
            volume_event = self.env.timeout(step_duration)
            yield volume_event
            volume -= step_duration * available_bandwidth
            if not initial_levels:
                initial_levels = cluster.get_levels()
            self.update_tier_on_move(source_tier, target_tier, step_duration * available_bandwidth,
                                     erase)
            self.register_step(t_start, step_duration, available_bandwidth, cluster,
                               target_tier, initial_levels, source_tier)
            if isinstance(source_tier, EphemeralTier):
                # if isinstance(target_tier, EphemeralTier):
                initial_levels = cluster.get_levels()
                eviction = source_tier.evict()
                # eviction = target_tier.evict()
                if eviction:
                    # register eviction step
                    self.register_step(t_start, step_duration, available_bandwidth, cluster,
                                       target_tier, initial_levels, source_tier, eviction)

        except simpy.exceptions.Interrupt:
            logger.trace(f'{self.appname} interrupt at {self.env.now}')
            step_duration = self.env.now - t_start
            if step_duration:
                self.last_event += step_duration
                volume -= step_duration * available_bandwidth
                if not initial_levels:
                    initial_levels = cluster.get_levels()
                self.update_tier_on_move(source_tier, target_tier,
                                         step_duration * available_bandwidth, erase)
                self.register_step(t_start, step_duration, available_bandwidth, cluster,
                                   target_tier, initial_levels, source_tier)
                if isinstance(source_tier, EphemeralTier):
                    initial_levels = cluster.get_levels()
                    eviction = source_tier.evict()
                    if eviction:
                        self.register_step(t_start, step_duration, available_bandwidth, cluster,
                                           target_tier, initial_levels, source_tier, eviction)

        return volume

    def evaluate_tier_bandwidth(self, cluster, tier):
        """Method to evaluate the bandwidth value for a given storage tier, and I/O operation, and a given I/O pattern.

        Args:
            cluster (Cluster): cluster object for which the bw will be evaluated.
            tier (Tier): the tier of the cluster storage system where the I/O will be executed.
        """
        assert isinstance(tier, Tier)
        # # assign bandwidth resource if not already done
        # self.env = env
        # if not tier.bandwidth:
        #     tier.bandwidth = BandwidthResource(IOPhase.current_ios, self.env, 10)
        logger.trace(f"bandwidth hardcoded : {self.bw}")
        logger.trace(f"operation: {self.operation} | pattern: {self.pattern}")
        if self.bw:
            available_bandwidth = self.bw
        else:
            logger.trace(f"Phases features: {self.phase_features}")
            max_bandwidth = cluster.get_max_bandwidth(
                tier,
                operation=self.operation,
                pattern=self.pattern,
                phase_features=self.phase_features)
            logger.trace(f"max bandwidth: {max_bandwidth}")
            self.bandwidth_concurrency = tier.bandwidth.count
            available_bandwidth = max_bandwidth/self.bandwidth_concurrency

        return available_bandwidth

    def get_step_duration(self, cluster, tier, volume):
        """Get the adequate step duration to not avoid I/O event or volume saturation in tier

        Args:
            cluster (Cluster): _description_
            tier (Tier): _description_
            volume (float): _description_

        Returns:
            tuple: _description_
        """

        available_bandwidth = self.evaluate_tier_bandwidth(cluster, tier)
        logger.trace(f"available bandwidth: {available_bandwidth}")
        # limit the volume to maximum available
        max_volume = min(volume, tier.capacity.capacity - tier.capacity.level)
        # take the smallest step, step_duration must be > 0
        if 0 < self.next_event - self.last_event < max_volume/available_bandwidth:
            step_duration = self.next_event - self.last_event
        else:
            step_duration = max_volume/available_bandwidth
        logger.trace(f"appname: {self.appname}, now : {self.env.now} | last event: {self.last_event} | next event: {self.next_event}"
                     f" | full duration: {volume/available_bandwidth} | step duration: {step_duration}")

        return step_duration, available_bandwidth

    def get_move_duration(self, cluster, source_tier, target_tier, volume):
        """Get the adequate step duration to avoid I/O event or volume saturation in tier

        Args:
            cluster (Cluster): _description_
            tier (Tier): _description_
            volume (float): _description_

        Returns:
            tuple (float, float): step duration and available bandwidth
        """

        read_from_source_bandwidth = cluster.get_max_bandwidth(source_tier, operation='read')/source_tier.bandwidth.count
        write_to_target_bandwidth = cluster.get_max_bandwidth(target_tier, operation='write')/target_tier.bandwidth.count
        bandwidths = [read_from_source_bandwidth, write_to_target_bandwidth]
        available_bandwidth = min(bandwidths)
        bottleneck_tier = source_tier if bandwidths.index(available_bandwidth) == 0 else target_tier
        # take the count of the bottleneck bandwidth
        self.bandwidth_concurrency = bottleneck_tier.bandwidth.count
        # limit the volume to available
        max_volume = min(volume, target_tier.capacity.capacity - target_tier.capacity.level)
        # take the smallest step, step_duration must be > 0
        if 0 < self.next_event - self.last_event < max_volume/available_bandwidth:
            step_duration = self.next_event - self.last_event
        else:
            step_duration = max_volume/available_bandwidth
        logger.trace(f"data-movement for: {self.appname}, now : {self.env.now} | "
                     f"last event: {self.last_event} | next event: {self.next_event}"
                     f" | full duration: {volume/available_bandwidth} | step duration: "
                     f"{step_duration}")

        return step_duration, available_bandwidth

    def move_step(self, env, cluster, source_tier, target_tier, erase=False):
        """Allows to run a movement of data in an interval where available bandwidth is constant.

        Args:
            env (simpy.Environment()): environment variable where the I/O operation will take place.
            cluster (Cluster): the cluster on which the application will run.
            source_tier (Tier): storage tier from which we read data to move.
            target_tier (Tier): storage tier where the data will be moved.

        Returns:
            bool: True if the step is completed, False otherwise.

        Yields:
            simpy.Event: other events that can occur during the I/O operation.
        """
        self.env = env
        if not source_tier.bandwidth:
            source_tier.bandwidth = BandwidthResource(IOPhase.current_ios, self.env, 10)
        if not target_tier.bandwidth:
            target_tier.bandwidth = BandwidthResource(IOPhase.current_ios, self.env, 10)

        volume = self.volume  # source_tier.capacity.level - 0.9*source_tier.capacity.capacity
        # retry data movement until its volume is consumed
        while volume > 0:
            with source_tier.bandwidth.request() as source_req:
                with target_tier.bandwidth.request() as target_req:
                    yield source_req & target_req
                    self.last_event = self.env.now
                    self.next_event = self.env.peek()

                    step_duration, available_bandwidth = self.get_move_duration(cluster,
                                                                                source_tier, target_tier, volume)
                    logger.trace(f"appname: {self.appname}, now : {self.env.now} | last event: "
                                 f"{self.last_event} | next event: {self.next_event}"
                                 f" | full duration: {volume/available_bandwidth} | step duration: {step_duration}")
                    initial_volumes = cluster.get_levels()
                    move_event = self.env.process(self.move_volume(step_duration, volume,
                                                                   available_bandwidth, cluster, source_tier, target_tier, erase=erase, initial_levels=initial_volumes))

                    # register the step event to be able to update it
                    IOPhase.current_ios.append(move_event)
                    # process the step volume
                    volume = yield move_event

        return True

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
        self.env = env
        # assign bandwidth resource if not already done
        if not tier.bandwidth:
            tier.bandwidth = BandwidthResource(IOPhase.current_ios, self.env, 10)
        volume = self.volume
        # retry IO until its volume is consumed
        while volume > 0:
            with tier.bandwidth.request() as req:
                yield req
                self.last_event = self.env.now
                self.next_event = self.env.peek()

                step_duration, available_bandwidth = self.get_step_duration(cluster, tier, volume)
                logger.trace(f"appname: {self.appname}, now : {self.env.now} | last event: {self.last_event} | next event: {self.next_event}"
                             f" | available bandwidth: {available_bandwidth}"
                             f" | full duration: {volume/available_bandwidth} | step duration: {step_duration}")
                initial_volumes = cluster.get_levels()
                step_event = self.env.process(self.process_volume(step_duration, volume,
                                                                  available_bandwidth, cluster, tier, initial_levels=initial_volumes))
                IOPhase.current_ios.append(step_event)
                # process the step volume
                volume = yield step_event

        return True

    def run(self, env, cluster, placement, use_bb=False, delay=0):
        """Allows to run an I/O operation."""
        self.env = env
        if delay:
            logger.info(f"(App {self.appname}) - Start delaying phase at {env.now}")
            yield self.env.timeout(delay)
            logger.info(f"(App {self.appname}) - End delaying phase at {env.now}")
        # get the tier where the I/O will be performed, if use_sbb=True, get the BB
        tier = get_tier(cluster, placement, use_bb=use_bb)
        # ret = yield self.env.process(self.run_step(self.env, cluster, tier))
        logger.info(f"(App {self.appname}) - Start I/O phase at {env.now}")
        if isinstance(tier, EphemeralTier):
            # retrieve EphemeralTier
            persistent_tier = get_tier(cluster, tier.persistent_tier)
            if self.operation == "read":
                # do prefetch
                io_prefetch = self.env.process(self.move_step(self.env, cluster,
                                                              persistent_tier,
                                                              #tier.persistent_tier,
                                                              tier,
                                                              erase=False))
                ret1 = yield io_prefetch
                io_event = self.env.process(self.run_step(self.env, cluster, tier))
                if ret1:
                    ret = yield io_event

            elif self.operation == "write":
                io_event = self.env.process(self.run_step(self.env, cluster, tier))
                # destage
                destage_event = self.env.process(self.move_step(self.env, cluster,
                                                                tier,
                                                                #tier.persistent_tier,
                                                                persistent_tier,
                                                                erase=False))
                # do not wait for the destage to complete
                # TODO, logic will fail if destaging is faster than the IO
                response = yield io_event | destage_event
                ret = all([value for key, value in response.items()])
        else:
            ret = yield self.env.process(self.run_step(self.env, cluster, tier))
        logger.info(f"(App {self.appname}) - End I/O phase at {env.now}")
        return ret


class RWIOPhase:
    """Represents a combined read/write I/O phase for an application."""

    def __init__(self, env, cores, volume_read, volume_write, pattern_read, pattern_write,
                 tier, data=None, appname=None, phase_features=None):
        """
        Initialize the RWIOPhase instance.

        Args:
            env (simpy.Environment): The simulation environment.
            cores (int): Number of compute cores used.
            volume_read (float): The volume of data to read.
            volume_write (float): The volume of data to write.
            pattern_read (float): The pattern of the read operation.
            pattern_write (float): The pattern of the write operation.
            tier (Tier): The storage tier where I/O operations will occur.
            data (simpy.Store, optional): Data store for application records.
            appname (str, optional): Name of the application.
            phase_features (PhaseFeatures, optional): Features of the I/O phase.
        """
        self.env = env
        self.cores = cores
        self.volume_read = volume_read
        self.volume_write = volume_write
        self.pattern_read = pattern_read
        self.pattern_write = pattern_write
        self.tier = tier
        self.data = data
        self.appname = appname
        self.phase_features = phase_features
        self.read_io = IOPhase(cores, 'read', volume_read, pattern_read,
                               data, appname)
        self.write_io = IOPhase(cores, 'write', volume_write, pattern_write,
                                data, appname)

        if self.phase_features is None:
            self.phase_features = PhaseFeatures(
                cores=self.cores, operation=self.operation,
                read_volume=self.read_volume,
                write_volume=self.write_volume,
                read_io_pattern=self.read_pattern,
                write_io_pattern=self.write_pattern,
                read_io_size=self.read_io_size,
                write_io_size=self.write_io_size)


class MixIOPhase():
    """Class that allows to run a mix of I/O operations. It initializes the read and write phases and registers processing steps in the data store with logging. It also updates tier levels with the algebraic value of volume and following a volume move. Finally, it runs a read-write I/O operation."""

    # TODO: add read_io_size=4e3 and  write_io_size=4e3 attributes and
    # TODO: pass phase_features object in cluster.get_max_bandiwdth

    def __init__(self, cores=1, read_volume=1e9, write_volume=1e9,
                 read_pattern=1, write_pattern=1,
                 read_io_size=4e3, write_io_size=4e3,
                 data=None, appname=None, read_bw=None,
                 write_bw=None):
        """Initialize the MixIO class."""
        self.cores = cores
        self.operation = "readwrite"
        # assert read_volume > 0, "read volume should be > 0 to initialize a MixIOPhase"
        self.read_volume = read_volume
        # assert write_volume > 0, "write volume should be > 0 to initialize a MixIOPhase"
        self.write_volume = write_volume
        self.read_pattern = read_pattern
        self.write_pattern = write_pattern
        self.read_io_size = read_io_size
        self.write_io_size = write_io_size
        self.read_bw = read_bw or None
        self.write_bw = write_bw or None
        self.data = data or None
        self.appname = appname or ''

        # TODO: Build here the PhaseFeatures object
        self.phase_features = PhaseFeatures(cores=self.cores,
                                            operation=self.operation,
                                            read_volume=self.read_volume,
                                            write_volume=self.write_volume,
                                            read_io_pattern=self.read_pattern,
                                            write_io_pattern=self.write_pattern,
                                            read_io_size=self.read_io_size,
                                            write_io_size=self.write_io_size)

        # initialize the read and write phases
        self.read_io = IOPhase(cores=cores, operation='read', volume=read_volume,
                               pattern=self.read_pattern, data=self.data, appname=self.appname, bw=read_bw, phase_features=self.phase_features)
        self.write_io = IOPhase(cores=cores, operation='write', volume=write_volume,
                                pattern=self.write_pattern, data=self.data, appname=self.appname, bw=write_bw, phase_features=self.phase_features)


    @property
    def volume(self):
        """Return the total volume of the mixed I/O operation."""
        return self.read_volume + self.write_volume

    def __str__(self):
        """Print in a human readable way a description of the mixed I/O Phase.

        Returns:
            string: description of I/O properties.
        """
        read_io_pattern = f"{self.read_pattern*100}% sequential | {(1-self.read_pattern)*100} % random"
        write_io_pattern = f"{self.write_pattern*100}% sequential | {(1-self.write_pattern)*100} % random"
        description = "-------------------\n"
        description += (f"{self.operation.capitalize()} I/O Phase of read volume {convert_size(self.read_volume)} with read pattern: {read_io_pattern} and write volume {convert_size(self.write_volume)} write pattern: {write_io_pattern}\n")
        logger.trace(description)
        return ""

    def register_step(self, t_start, step_duration, available_bandwidth, cluster, tier,
                      initial_levels=None, source_tier=None, eviction=None):
        """Registering a processing step in the data store with some logging.

        Args:
            t_start (float): timestamp of the start of the step.
            step_duration (float): duration of the step.
            available_bandwidth (float): available bandwidth in the step.
            cluster (Cluster): the cluster on which the phase will run.
            tier (Tier): the tier on which the step will run.
            initial_levels (dict): initial levels of all tiers at the start of the step.
            source_tier (Tier, optional): the tier from which the step will run.
            eviction (int, optional): volume of data which was evicted from ephemeral tier.
        """

        self.read_io.register_step(t_start, step_duration, available_bandwidth, cluster, tier,
                                   initial_levels=initial_levels, source_tier=source_tier, eviction=eviction)
        self.write_io.register_step(t_start, step_duration, available_bandwidth, cluster, tier,
                                    initial_levels=initial_levels, source_tier=source_tier, eviction=eviction)

    def update_tier(self, tier, volume):
        """Update tier level with the algebric value of volume.

        Args:
            tier (Tier): tier for which the level will be updated.
            volume (float): volume amount (positive or negative) to adjust tier level.
        """
        self.read_io.update_tier(tier, volume)
        self.write_io.update_tier(tier, volume)

    def update_tier_on_move(self, source_tier, target_tier, volume, erase):
        """Update tier level following a volume move.

        Args:
            source_tier (Tier): tier from which the data will be moved.
            target_tier (Tier): tier for which the level will be updated.
            volume (float): volume value (positive or negative) to adjust tier level.
            erase (bool): whether or not erase the amount of volume from source_tier.
        """
        self.read_io.update_tier_on_move(source_tier, target_tier, volume, erase)
        self.write_io.update_tier_on_move(source_tier, target_tier, volume, erase)

    def run(self, env, cluster, placement, use_bb=False, delay=0):
        """Run an RW I/O operation."""
        self.env = env
        if delay:
            logger.info(f"(App {self.appname}) - Start delay phase at {env.now}")
            yield self.env.timeout(delay)
            logger.info(f"(App {self.appname}) - End delay phase at {env.now}")
        # get the tier where the I/O will be performed, if use_sbb=True, get the BB
        tier = get_tier(cluster, placement, use_bb=use_bb)
        # ret = yield self.env.process(self.run_step(self.env, cluster, tier))
        # initializing events
        # read_event = Event()
        # write_event = Event()
        logger.info(f"(App {self.appname}) - Start RW I/O phase at {env.now}")
        if isinstance(tier, EphemeralTier):
            # if self.read_volume > 0:
            # do prefetch for ephemeral tier
            # NOTE: when tier is ephemeral, tier_persistent_tier is string
            tier_persistent = get_tier(cluster, tier.persistent_tier)
            io_read_prefetch = self.env.process(
                self.read_io.move_step(self.env, cluster, tier_persistent,
                                       tier, erase=False))
            ret1 = yield io_read_prefetch
            io_read_event = self.env.process(
                self.read_io.run_step(self.env, cluster, tier))
            # if self.write_volume > 0:
            io_write_event = self.env.process(
                self.write_io.run_step(self.env, cluster, tier))
            # destage
            destage_event = self.env.process(
                self.write_io.move_step(self.env, cluster, tier,
                                        tier_persistent, erase=False))
            # TODO: should prefetech starts in the same time as the read?
            # Default is sequential, the write should begin simultaneously with read as well as destaging
            if ret1:
                ret2 = yield io_read_event & io_write_event & destage_event

            # do not wait for the destage to complete
            # TODO: logic will fail if destaging is faster than the IO
            response = yield AllOf(self.env, (io_read_prefetch, io_read_event)) & AnyOf(self.env, (io_write_event, destage_event))
            ret = all(value for key, value in response.items())
        else:
            # persistent tier
            read_io_event = self.env.process(self.read_io.run_step(self.env, cluster, tier))
            write_io_event = self.env.process(self.write_io.run_step(self.env, cluster, tier))

            response = yield AllOf(self.env, (read_io_event, write_io_event))
            ret = all([value for key, value in response.items()])
        logger.info(f"(App {self.appname}) - End RW I/O phase at {env.now}")
        return ret


# def __init__(self, cores=1, operation='read', volume=1e9, pattern=1,
#                  data=None, appname=None, bw=None):
#         """Inits an instance of I/O phase."""
#         self.cores = cores
#         self.operation = operation
#         assert self.operation in ['read', 'write']
#         self.volume = volume
#         self.pattern = pattern
#         self.last_event = 0
#         self.next_event = 0
#         self.bandwidth_concurrency = 1
#         self.data = data or None
#         self.appname = appname or ''

#         # Bytes to MB conversion
#         # case where bandwidth is given for a reproducing a simulation
#         self.bw = bw*1e6 if bw else bw
#         # logger.info(self.__str__())

# def process_volume(self, step_duration, volume, available_bandwidth, cluster, tier,
    #                    initial_levels=None):
    #     """This method processes a small amount of I/O volume between two predictable events on a specific tier. If an event occurs in the meantime, I/O will be interrupted and bandwidth updated according.

    #     Args:
    #         step_duration (float): the expected duration between two predictable events.
    #         volume (float): volume in bytes of the data to process.
    #         cluster (Cluster): cluster facility where the I/O operation should take place.
    #         available_bandwidth (float): available bandwidth in the step.
    #         tier (Tier): storage tier concerned by the I/O operation. It could be reading from this tier or writing to it.
    #     """
    #     read_volume = self.read_io.process_volume(step_duration, volume, available_bandwidth, cluster, tier)

    # def evaluate_tier_bandwidth(self, cluster, tier):
    #     """Method to evaluate the bandwidth value for a given storage tier, and I/O operation including readwrite, and a given I/O pattern.

    #     Args:
    #         cluster (Cluster): cluster object for which the bw will be evaluated.
    #         tier (Tier): the tier of the cluster storage system where the I/O will be executed.
    #     """
    #     assert isinstance(tier, Tier)
    #     # # assign bandwidth resource if not already done
    #     # self.env = env
    #     # if not tier.bandwidth:
    #     #     tier.bandwidth = BandwidthResource(IOPhase.current_ios, self.env, 10)
    #     if self.read_bw and self.write_bw:
    #         available_bandwidth = self.read_bw, self.write_bw
    #     else:
    #         read_max_bandwidth = cluster.get_max_bandwidth(tier, operation="read", pattern=self.read_pattern)
    #         write_max_bandwidth = cluster.get_max_bandwidth(tier, operation="write", pattern=self.write_pattern)
    #         self.bandwidth_concurrency = tier.bandwidth.count
    #         available_bandwidth = read_max_bandwidth/self.bandwidth_concurrency, write_max_bandwidth/self.bandwidth_concurrency

    #     return available_bandwidth
