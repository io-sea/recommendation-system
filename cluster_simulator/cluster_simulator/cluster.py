#!/usr/bin/env python
"""
This module proposes a class to define HPC cluster as a set of compute resources and storage facilities. Both are shared between applications running on the cluster.
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
import math
from cluster_simulator.utils import convert_size, BandwidthResource



def convert_size(size_bytes):
    """Function to display a data volume in human readable way (B, KB, MB,...) instead of 1e3, 1e6, 1e9 bytes.

    Args:
        size_bytes (float): volume of data in bytes to convert.

    Returns:
        string: containing the volume expressed with a more convenient unit.
    """
    BYTE_UNIT = 1000  # replace 1000 by 1024 fir B, KiB, MiB, ...
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, BYTE_UNIT)))
    p = math.pow(BYTE_UNIT, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"  # "%s %s" % (s, size_name[i])


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


class Cluster:
    """A cluster is a set of compute nodes each node having a fixed number of cores. The storage system is heterogenous and consists on a set of tiers. Each tier has its own capacity and performances.
    Generally speaking, these tiers are referenced as persistent storage, which means that data conveyed through tiers is kept persistently and can be retrieved anytime by an application, unless it is explicitly removed as in data movers methods.
    A cluster contains also a specific type of tiers that is ephemeral, which means that it does not keep data beyond the execution time of an application. Ephemeral tiers are often supported by datanodes that hold their own compute units and also high storage hardware to serve as burst buffers backend. Burst buffers partitions their resources into flavors to dispatch them smartly between applications. They have also their own policy of eviction when storage are (quasi) saturated as well as a destaging capacity in order to move data to a persistent storage tier. As a consequence, each ephemeral tier has its a specific tier attached to it.
    """

    def __init__(self, env, compute_nodes=1, cores_per_node=2, tiers=[], ephemeral_tier=None,
                 data=None):
        """Inits a Cluster instance with mentioned attributes.

        Args:
            env (simpy.Environment): an environement where all simulation happens.
            compute_nodes (int, optional): number of compute nodes to specify. Defaults to 1.
            cores_per_node (int, optional): number of cores per compute node. Defaults to 2.
            tiers (list, optional): list of instances from Tier class. Defaults to [].
            ephemeral_tier (EphemeralTier, optional): specifies an ephemeral tier attached to this cluster. Defaults to None.
            data (simpy.Store, optional): a queue like object to store data relative to execution of an application on the cluster. Defaults to None.
        """
        self.compute_nodes = simpy.Resource(env, capacity=compute_nodes)
        self.compute_cores = simpy.Resource(env, capacity=cores_per_node*compute_nodes)
        self.data = data or None
        self.tiers = [tier for tier in tiers if isinstance(tier, Tier)]
        self.ephemeral_tier = ephemeral_tier if isinstance(ephemeral_tier, EphemeralTier) else None
        # logger.info(self.__str__())

    def __str__(self):
        """Displays some cluster properties in readable way.

        Returns:
            string: description of the cluster properties and state.
        """
        description = "====================\n"
        description += (f"Cluster with {self.compute_nodes.capacity} compute nodes \n")
        description += f"Each having {self.compute_cores.capacity} cores in total \n"
        if self.tiers:
            for tier in self.tiers:
                description += tier.__str__()
        return description

    def get_levels(self):
        """Gathers tiers levels snapshot at a specific time event into a dict.

        Returns:
            levels (dict): snapshot of the tiers levels.
        """
        levels = dict([(tier.name, tier.capacity.level) for tier in self.tiers])
        if self.ephemeral_tier:
            levels[self.ephemeral_tier.name] = self.ephemeral_tier.capacity.level
        return levels

    def get_max_bandwidth(self, tier, cores=1, operation='read', pattern=1):
        """Get the maximum bandwidth for a given tier, number of cores dedicated to the operation, a type of operation. Sequential pattern are assumed during copy/move as well as an important blocksize.

        Args:
            tier (Tier or index): the tier from which the bandwidth will be estimated.
            cores (int, optional): _description_. Defaults to 1.
            operation (str, optional): 'read' or 'write'. Defaults to 'read'.
            pattern (int, optional): 1 for seq and 0 for random. Defaults to 1.

        Returns:
            float: a bandwidth value in MB/s for the specified arguments.
        """
        if not isinstance(tier, Tier):
            tier = get_tier(self, tier)
        return (tier.max_bandwidth[operation]['seq'] * pattern +
                tier.max_bandwidth[operation]['rand'] * (1-pattern)) * cores * 1e6


class Tier:
    """Model a tier storage service with a focus on a limited bandwidth resource as well as a limited capacity. In this model we expect a bandwidth value at its asymptotic state, so blocksize is still not a parameter. Only the asymptotic part of the throughput curve is considered. Other considered variables are read/write variables and sequential/random variables.
    Output is a scalar value in MB/s.
    Typically we access the bandwidth value as in dictionary: b['read']['seq'] = 200MB/s.
    # TODO: extend this to a NN as function approximator to allow:
        averaging over variables
        interpolation when data entry is absent, i.e. b['seq'] gives a value

    """

    def __init__(self, env, name, bandwidth, capacity=100e9):
        """Inits a tier instance with some storage service specifications.

        Args:
            env (simpy.Environment): the simpy environment where all simulations happens.s
            name (string): a name to make analytics readable and to assign a unique ID to a tier.
            bandwidth (simpy.Resource): bandwidth as limited number of slots that can be consumed.  Default capacity is up to 10 concurrent I/O sharing the maximum value of the bandwidth.
            capacity (simpy.Container, optional): storage capacity of the tier. Defaults to 100e9 (100GB).
            max_bandwidth (dict): which contains operation and pattern dependant max bandwidths.
            Example:
                ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
                         'write': {'seq': 100, 'rand': 100}}
        """
        self.env = env
        self.name = name
        self.capacity = simpy.Container(self.env, init=0, capacity=capacity)
        self.max_bandwidth = bandwidth
        # modeling percent use of bandwidth
        #self.bandwidth = simpy.Resource(self.env, capacity=10)
        #self.bandwidth = BandwidthResource(self.env, event_list=None, capacity=10)
        self.bandwidth = None
        self.bandwidth_concurrency = dict()
        # self.bandwidth = simpy.Container(env, init=100, capacity=100)
        # logger.info(self.__str__())

    def __str__(self):
        """Prints cluster related information in a human readable fashion.

        Returns:
            string: properties and state of the cluster.
        """
        description = "\n-------------------\n"
        description += (f"Tier: {self.name} with capacity = {convert_size(self.capacity.capacity)}\n")
        description += ("{:<12} {:<12} {:<12}".format('Operation', 'Pattern', 'Bandwidth MB/s')+"\n")
        for op, inner_dict in self.max_bandwidth.items():
            for pattern, value in inner_dict.items():
                description += ("{:<12} {:<12} {:<12}".format(op, pattern, value)+"\n")
        return description


class EphemeralTier(Tier):
    """Ephemeral tiers are tiers that are used for the duration of an application or a workflow. They are attached to a persistent Tier where data will resided at the end of the application or workflow.
    When the app should read data from a tier 1, and will make use of a transient tier, this data can be prefetched to the transient tier to be accessed later by the app from ephemeral tier.
    When the app should write data to a target tier 2, and will make use of a transient tier t, this data will be first written to tier t, and then when destaging policy is triggered destage data to tier 2.

    Args:
        persistent_tier(Tier): persistent tier attached to this transient/ephemeral tier where all data conveyed by the application will be found.
    """

    def __init__(self, env, name, persistent_tier, bandwidth, capacity=80e9):
        """Inits an EphemeralTier instance.

        Args:
            env (simpy.Environment): the simpy environment where all simulations happens.s
            name (string): unique name
            persistent_tier (Tier): a persistent tier the burst buffer is attached to. Will be automatically used for destaging purposes.
            bandwidth (simpy.Resource): bandwidth as limited number of slots that can be consumed.  Default capacity is up to 10 concurrent I/O sharing the maximum value of the bandwidth.
            capacity (simpy.Container, optional): storage capacity of the flavor of the datanode. Defaults to 80e9.
        """
        self.env = env
        # the non transient tier it is attached to
        self.persistent_tier = persistent_tier
        # amount of dirty data
        self.dirty = 0
        # eviction parameters
        self.lower_threshold = 0.7
        self.upper_threshold = 0.9
        # self.cores = simpy.Resource(env, capacity=cores)
        super().__init__(env, name, bandwidth, capacity=capacity)

    def evict(self):
        """Check if the application should evict some data from the ephemeral tier.

        Args:
            ephemeral_tier (EphemeralTier): the ephemeral tier where data is stored.
            lower_threshold (float): lower threshold for the eviction.
            upper_threshold (float): upper threshold for the eviction.

        Returns:
            eviction_volume (int): amount of data to be evicted.
        """
        if self.capacity.level > self.upper_threshold*self.capacity.capacity:
            clean_data = self.capacity.level - self.dirty
            eviction_volume = min(self.capacity.level -
                                  self.lower_threshold*self.capacity.capacity, clean_data)
            if eviction_volume > 0:
                self.capacity.get(eviction_volume)
                logger.info(f"{convert_size(eviction_volume)} evicted from tier {self.name}")
                return eviction_volume
        return 0


def bandwidth_share_model(n_threads):
    """Description of a bandwidth share model that could extend the behavior from storage services measurements.

    Args:
        n_threads (int): number of threads/processes processing I/O simultaneously.

    Returns:
        float: the bandwidth share of the last process.
    """
    return np.sqrt(1 + n_threads)/np.sqrt(2)


def compute_share_model(n_cores):
    """Description of parallelizing compute resources for an application. The gain factor is considered for a reference duration when using a single unit of computing.

    Args:
        n_cores (int): number of cores (computing unit) the application is distributed on.

    Returns:
        float: the speedup factor in comparison when using a single compute unit.
    """
    return np.sqrt(1 + n_cores)/np.sqrt(2)


def get_tier(cluster, tier_reference, use_bb=False):
    """Cluster has attributes called tiers and ephemeral_tier. The first one is a list and the second one is a single object attached to one of the(persistent) tiers.
    Reference to a tier could be either a string(search by name) or an integer(search by index in the list of tiers).
    When a placement refers to a Tier object, and use_bb is False, data will be placed in the tier.
    When use_bb is True, data will be placed in the ephemeral_tier which is attached to the indicated tier.

    Args:
        cluster (Cluster): a cluster object that contains the tiers
        tier_reference (string, int): tier name or index in the list of tiers
        use_bb (bool): if True, data will be placed in the ephemeral_tier which is attached to the indicated tier.

    Returns:
        Tier or EphemeralTier: The storage tier that will be targeted for I/O operations.
    """
    def find_tier(cluster, tier_reference):
        """ Nested function to retrieve tier by its reference.
        Returns:
            tier(Tier): the tier object that was referenced."""
        tier = None
        if isinstance(tier_reference, int):
            tier = cluster.tiers[tier_reference]
        elif isinstance(tier_reference, str):
            for cluster_tier in cluster.tiers:
                if tier_reference == cluster_tier.name:
                    tier = cluster_tier

        else:  # tier_reference is an instance of Tier
            # TODO: reactivate the None clause
            tier = tier_reference
        if tier is None:
            raise ValueError(f"Tier {tier} not found")

        return tier
    if not use_bb:
        return find_tier(cluster, tier_reference)
    if use_bb:
        return cluster.ephemeral_tier
