#!/usr/bin/env python
"""
This module proposes a class to define HPC applications as a sequence of phases that are compute, read or write dominant behavior.
"""


__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
""" = """
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
from cluster_simulator.cluster import Cluster, Tier, bandwidth_share_model, compute_share_model, get_tier
from cluster_simulator.phase import DelayPhase, ComputePhase, IOPhase
import copy
from cluster_simulator.utils import name_app, convert_size
from simpy.events import AnyOf, AllOf, Event
from loguru import logger
import math


class Application:
    """Defining an application as a sequential set of phases(Compute/Read/Write) that occur in linear fashion. Next phase cannot proceed until previous one is finished.

    Each phases has its own attributes and resources dedicated to it. But all phases need computes units(called here cores) in order to run.

    Special attention is given to phases compute units: when phases requests different number of cores the application will wait for the availability of the maximum requested cores. Once available within the cluster, this max number if locked until the application execution is finished.

    Attributes:
        env: the environment object where all the discrete event simulation occurs.
        name: a string name can be given to the application that is visible in plotting utilities and logging traces. If not provided, a random string will be given the the application.
        compute: a list of any size that contains times where events will occur, and compute phases are between the elements of the list.
        read: a list of the same length as compute, for each timestamp indicated in the compute list the read list indicates the volume of data in bytes that will be read during this phase. The duration of the read phase depends on the cluster hardware.
        write: a list of the same length as compute. For each time indicated in the compute list, the write list contains the volume of bytes that should be written by the application at this timed event. The duration of the write phase depends on the cluster hardware.
        data: (optional) a simpy.Store object that stores the phases schedule of the application and make it available outside of the application.
        delay: time in seconds to wait in order to start the application.

    Applications running on the same cluster are stackables as well as their relative phases and thus can run in parallel. A start delay can be applied to any application that postpones its scheduling.

    """

    def __init__(self, env, name=None, compute=None, read=None,
                 write=None, bw=None, data=None, delay=0):
        """Initialize the application and schedule the sequence of phases.

        Args:
            env:
                simpy object relative the discrete event simulation environment.
            name:
                string to indicate the reference name of the application.
            store:
                simpy.Store object containing sequence of phases
            compute:
                a list of any size that contains times where events will occur, and compute phases are between the elements of the list.
            read:
                a list of the same length as compute, for each timestamp indicated in the compute list the read list indicates the volume of data in bytes that will be read during this phase. The duration of the read phase depends on the cluster hardware.
            write:
                a list of the same length as compute. For each time indicated in the compute list, the write list contains the volume of bytes that should be written by the application at this timed event. The duration of the write phase depends on the cluster hardware.
            data:
                (optional) a simpy.Store object that stores the phases schedule of the application and make it available outside of the application.
            bw :
                (list, optional) in MB/s the observed throughput for this IO to reproduce the observed results.
            delay:
                time in seconds to wait in order to start the application.
        """
        self.env = env
        self.name = name or name_app()
        self.store = simpy.Store(self.env)
        self.compute = compute
        self.read = read
        self.write = write
        self.delay = delay
        # ensure format is valid, all list are length equal
        assert all([len(lst) == len(self.compute) for lst in [self.read, self.write]])
        self.data = data or None
        self.bw = bw or None
        self.status = None
        # schedule all events
        self.cores_request = []
        self.schedule()

    def put_delay(self, duration):
        """Add a Delay phase that waits before starting the application. This phase consumes 0 units of compute resources. It subclasses DeplayPhase and the object is queued to store attribute and cores needed to cores_request list.

        Args:
            duration (float):
                time in seconds of the delay phase.
        """
        delay_phase = DelayPhase(duration, data=self.data, appname=self.name)
        self.store.put(delay_phase)
        self.cores_request.append(0)

    def put_compute(self, duration, cores=1):
        """Add a compute phase that requests some unit cores and lasts a specific duration. It subclasses ComputePhase and object is queued to store attribute and cores needed to cores_request list.

        Args:
            duration (float):
                time in seconds of the compute phase.
            cores (int):
                number of cores
        """
        compute_phase = ComputePhase(duration, cores, data=self.data, appname=self.name)
        self.store.put(compute_phase)
        self.cores_request.append(compute_phase.cores)

    def put_io(self, operation, volume, pattern=1, bw=None):
        """Add an I/O phase in read or write mode with a specific volume and pattern. It subclasses IOPhase and the object is queued to store attribute.

        Args:
            operation (string):
                type of I/O operation, "read" or "write". Cannot schedule a mix of both.
            volume (float): volume in bytes of data to be processed by the I/O.
            pattern (float):
                encodes sequential pattern for a value of 1, and a random pattern for value of 0. Accepts intermediate values like 0.2, i.e. a mix of 20% sequential and 80% random. Default value is set to 1.
            bw (float):
                if note None initiates an IOPhase with bw argument to comply to mandatory observed bandwidth.
        """
        bw = bw or None
        io_phase = IOPhase(cores=1, operation=operation, volume=volume,
                           pattern=pattern, data=self.data, bw=bw, appname=self.name)
        self.store.put(io_phase)
        self.cores_request.append(io_phase.cores)

    def schedule(self):
        """Read the compute/read/write inputs from application attributes and schedule them in a sequential order.

        Args:
            status (list of bool): store the sequential status of each element of the application sequence.
        """
        self.status = []
        if self.delay > 0:
            self.put_delay(duration=self.delay)
            self.status.append(False)
        for i, _ in enumerate(self.compute):
            # iterating over timestamps
            if self.read[i] > 0:
                # register read phases
                phase_bw = self.bw[i] if self.bw else None
                self.put_io(operation="read", volume=self.read[i], bw=phase_bw)
                self.status.append(False)
            if self.write[i] > 0:
                # register write phases
                phase_bw = self.bw[i] if self.bw else None
                self.put_io(operation="write", volume=self.write[i], bw=phase_bw)
                self.status.append(False)
            if i < len(self.compute) - 1:
                # register compute phase with duration  = diff between two events
                duration = self.compute[i+1] - self.compute[i]
                self.put_compute(duration, cores=1)
                self.status.append(False)

    def request_cores(self, cluster):
        """Issues a request on compute cores to get a slot as a shared resources. Takes the maximum of requested cores through the application phases as a reference resource amount to lock for the application duration.

        Args:
            cluster (Cluster): accepts an object of type Cluster that contains the compute resources.
        Returns:
            an array of requests of each individual compute unit (core).
        """
        return [cluster.compute_cores.request() for _ in range(max(self.cores_request))]

    def run(self, cluster, placement, use_bb=None):
        """Launches the execution process of an application on a specified cluster having compute and storage resources with placement indications for each issued I/O. Phases are executed sequentially.

        Args:
            cluster (Cluster): a set of compute resources with storage services.
            tiers (_type_): _description_

        Returns:
            data (simpy.Store): list of objects that stores scheduled phases of the application.

        Yields:
            simpy.Event: relative events.
        """
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
                data_placement = cluster.tiers[int(placement[item_number])]
                bb = use_bb[item_number] if use_bb else False
                if phase == 0:
                    yield AllOf(self.env, requesting_cores)
                    ret = yield self.env.process(item.run(self.env, cluster, placement=data_placement, use_bb=bb))
                    self.status[phase] = ret
                    logger.debug(f"the issued status of the IO phase : {self.status[phase]}")
                elif phase > 0 and self.status[phase-1] == True:
                    self.status[phase] = yield self.env.process(item.run(self.env, cluster, placement=data_placement, use_bb=bb))
                else:
                    self.status[phase] = False
                phase += 1
                item_number += 1

        releasing_cores = [cluster.compute_cores.release(core) for core in requesting_cores]
        yield AllOf(self.env, releasing_cores)
        return self.data

    def get_fitness(self):
        """Method to get execution duration of the applications. It iterate over records saved in data to find the phase having the latest timestamp.

        Record example:
        sample_item = {'app': 'B8', 'type': 'read', 'cpu_usage': 1, 't_start': 0, 't_end': 4.761904761904762, 'bandwidth': 210.0, 'phase_duration': 4.761904761904762, 'volume': 1000000000.0, 'tiers': ['SSD', 'NVRAM'], 'data_placement': {'placement': 'SSD'}, 'tier_level': {'SSD': 1000000000.0, 'NVRAM': 0}}

        Returns:
            float: the timestamp of the last event of the session.
        """
        t_min = math.inf
        t_max = 0
        if not self.data:
            logger.error("No data store provided")
            return None
        if not self.data.items:
            return t_max - t_min
        for phase in self.data.items:
            if phase["type"] in ["read", "write", "compute"]:
                t_max = max(t_max, phase["t_end"])
                t_min = min(t_min, phase["t_start"])
        return t_max - t_min

    def get_ephemeral_size(self):
        """Method to get the maximum space used by an ephemeral tier. It iterate over records saved in data to find the if a tier is in tier_level but not in tiers and get its maximum level.

        Record example:
        | Monitoring| app: 1 | type: read | cpu_usage: 1 | t_start: 37.5 | t_end: 46.0 | bandwidth_concurrency: 2 | bandwidth: 40.0 MB/s | phase_duration: 8.5 | volume: 340.0 MB | tiers: ['HDD'] | data_placement: {'placement': 'HDD'} | init_level: {'HDD': 2170000000.0, 'BB': 1700000000.0} | tier_level: {'HDD': 2170000000.0, 'BB': 1700000000.0} | BB_level: 1700000000.0

        Returns:
            float: the maximum level reached by the ephemeral tier.
        """
        if not self.data:
            logger.error("No data store provided")
            return None
        if not self.data.items:
            return None
        bb = list(set(self.data.items[0]["tier_level"].keys()) - set(self.data.items[0]["tiers"]))
        if bb:
            ephemeral_tier = bb[0]
            max_level = 0
            for phase in self.data.items:
                max_level = max(max_level, phase["tier_level"][ephemeral_tier])
            return max_level
        else:
            return None



