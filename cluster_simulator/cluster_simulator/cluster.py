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
import os
from loguru import logger
import numpy as np
import pandas as pd
import math
import joblib
import yaml
import joblib
from cluster_simulator.utils import convert_size, BandwidthResource
from cluster_simulator.phase_features import PhaseFeatures
from cluster_simulator import DEFAULT_CONFIG_PATH
from performance_data.data_model import load_and_predict
from dataclasses import dataclass, field
from enum import Enum



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

def read_yaml(input_data):
    # Check if the input is a file path
    if isinstance(input_data, str) and os.path.isfile(input_data):
        with open(input_data, 'r') as file:
            data = yaml.safe_load(file)
    # Check if the input is a YAML data string
    elif isinstance(input_data, str):
        data = yaml.safe_load(input_data)
    else:
        raise ValueError("Invalid input. Please provide a YAML file path or a YAML data string.")

    return data


class Cluster:
    """A cluster is a set of compute nodes each node having a fixed number of cores. The storage system is heterogenous and consists on a set of tiers. Each tier has its own capacity and performances.
    Generally speaking, these tiers are referenced as persistent storage, which means that data conveyed through tiers is kept persistently and can be retrieved anytime by an application, unless it is explicitly removed as in data movers methods.
    A cluster contains also a specific type of tiers that is ephemeral, which means that it does not keep data beyond the execution time of an application. Ephemeral tiers are often supported by datanodes that hold their own compute units and also high storage hardware to serve as burst buffers backend. Burst buffers partitions their resources into flavors to dispatch them smartly between applications. They have also their own policy of eviction when storage are (quasi) saturated as well as a destaging capacity in order to move data to a persistent storage tier. As a consequence, each ephemeral tier has its a specific tier attached to it.


    Args:
        env (simpy.Environment): The SimPy environment.
        config_path (str, optional): Path to the YAML configuration file.
        compute_nodes (int, optional): Number of compute nodes. Overrides the value
            specified in the YAML file if provided.
        cores_per_node (int, optional): Number of cores per node. Overrides the value
            specified in the YAML file if provided.
        tiers (list of Tier instances, optional): List of Tier instances. Overrides
            the value specified in the YAML file if provided.
        ephemeral_tier (Tier instance, optional): The ephemeral storage tier. Overrides
            the value specified in the YAML file if provided.

    Raises:
        ValueError: If there is a mismatch between tier names in the provided
            list of Tier instances and the YAML file.

    Attributes:
        env (simpy.Environment): The SimPy environment.
        compute_nodes (int): Number of compute nodes.
        compute_cores (int): Total number of cores in the cluster.
        tiers (list of Tier instances): List of storage tiers in the cluster.
        ephemeral (Tier instance): The ephemeral storage tier, if any.

    Methods:
        __str__(): Returns a string representation of the cluster.
        get_tier_by_name(name): Returns the Tier instance with the specified name.
        get_tier_by_path(path): Returns the Tier instance with the specified
            bandwidth model path.
        set_ephemeral_tier(tier): Sets the ephemeral storage tier.

    Example:
        # Create a cluster with a custom number of compute nodes and cores per node.
        cluster = Cluster(env, config_path='config.yaml', compute_nodes=4, cores_per_node=8)

        # Override the tiers specified in the YAML file with a custom list of Tier instances.
        tiers = [Tier(env, 'tier1', 100e9), Tier(env, 'tier2', 500e9)]
        cluster = Cluster(env, config_path='config.yaml', tiers=tiers)
    """

    def __init__(self, env, config_path=None, compute_nodes=1, cores_per_node=2, tiers=None,
                 ephemeral_tier=None):
        """Initializes a Cluster instance.

        Args:
            env (simpy.Environment): The simpy environment where all simulation happens.
            config_path (str, optional): The path to the YAML configuration file to use. Defaults to DEFAULT_CONFIG_PATH.
            compute_nodes (int, optional): The number of compute nodes to specify. Overrides the 'compute_nodes' value in the YAML configuration file if provided. Defaults to 1.
            cores_per_node (int, optional): The number of cores per compute node. Overrides the 'cores_per_node' value in the YAML configuration file if provided. Defaults to 2.
            tiers (list, optional): A list of instances from the Tier class to use. Overrides the 'tiers' value in the YAML configuration file if provided. Defaults to [].
            ephemeral_tier (Tier, optional): Specifies an ephemeral tier attached to this cluster. Overrides the 'ephemeral_tier' value in the YAML configuration file if provided. Defaults to None.

        """
        logger.debug("Initializing Cluster instance")
        self.env = env
        self.compute_nodes = simpy.Resource(env, capacity=compute_nodes)
        self.compute_cores = simpy.Resource(env, capacity=cores_per_node*compute_nodes)
        self.tiers = tiers or []
        self.ephemeral_tier = ephemeral_tier

        if config_path:
            logger.debug(f"Loading configuration from {config_path}")
            # with open(config_path) as f:
            #     config = yaml.safe_load(f)
            config = read_yaml(config_path)

            # Override default values if provided in YAML file
            default_config = config.get('defaults')
            if default_config:
                compute_nodes = default_config.get('compute_nodes')
                cores_per_node = default_config.get('cores_per_node')
                self.compute_nodes = simpy.Resource(env,
                                                    capacity=compute_nodes)
                self.compute_cores = simpy.Resource(env,
                                                    capacity=cores_per_node*compute_nodes)

            # Create tiers from configuration
            if tiers is None:
                logger.debug("Creating tiers from configuration")
                self.tiers = []
                for tier_cfg in config['tiers']:
                    tier_name = tier_cfg['name']
                    tier_capacity = int(tier_cfg['capacity'])
                    tier_max_bandwidth = tier_cfg.get('max_bandwidth')
                    tier_bandwidth_model_path = tier_cfg.get('bandwidth_model_path')

                    if tier_max_bandwidth is None and tier_bandwidth_model_path is None:
                        raise ValueError('Either max_bandwidth or model_path is mandatory')

                    logger.debug(f"[Yaml Parsing] tier: {tier_name} | capacity: {tier_capacity} | tier bandwidth: {tier_max_bandwidth} | tier model: {tier_bandwidth_model_path}")
                    tier = Tier(env=self.env, name=tier_name,
                                max_bandwidth=tier_max_bandwidth, bandwidth_model_path=tier_bandwidth_model_path, capacity=tier_capacity)
                    self.tiers.append(tier)
            else:
                logger.debug("Overriding tiers with provided values")
                self.tiers = tiers

            if ephemeral_tier is None:
                ephemeral_cfg = config.get('ephemeral_tier')
                if ephemeral_cfg:
                    logger.debug("Ephemeral tier")
                    # (self, env, name, persistent_tier, max_bandwidth, capacity=80e9
                    ephemeral_tier = EphemeralTier(self.env,
                                                   ephemeral_cfg['name'],
                                                   persistent_tier=ephemeral_cfg['persistent_tier'],
                                                   max_bandwidth=ephemeral_cfg['max_bandwidth'],
                                                   capacity=ephemeral_cfg['capacity'])
                    logger.debug(f"[Yaml Parsing] tier: {ephemeral_tier.name} | capacity: {ephemeral_tier.capacity.capacity} | tier bandwidth: {ephemeral_tier.max_bandwidth} | tier model: {ephemeral_tier.bandwidth_model_path}")
                    self.ephemeral_tier = ephemeral_tier
            else:
                logger.debug("Overriding ephemeral_tier with provided values")
                self.ephemeral_tier = ephemeral_tier

    def __str__(self):
        """Displays some cluster properties in readable way.

        Returns:
            string: description of the cluster properties and state.
        """
        description = "\n====================\n"
        description += (f"Cluster with {self.compute_nodes.capacity} compute nodes \n")
        description += f"Each having {self.compute_cores.capacity} cores in total \n"
        if self.tiers:
            for tier in self.tiers:
                description += tier.__str__()
        if self.ephemeral_tier:
            description += self.ephemeral_tier.__str__()
        logger.trace(description)
        return ""

    def get_levels(self):
        """Gathers tiers levels snapshot at a specific time event into a dict.

        Returns:
            levels (dict): snapshot of the tiers levels.
        """
        levels = {tier.name: tier.capacity.level for tier in self.tiers}
        if self.ephemeral_tier is not None:
            levels[self.ephemeral_tier.name] = self.ephemeral_tier.capacity.level
        return levels

    def get_max_bandwidth(self, tier, cores=1, operation='read', pattern=1,
                          phase_features=None):
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

        if tier.bandwidth_model_path:
            if phase_features is None:
                phase_features = PhaseFeatures(cores=cores,
                                               operation=operation,
                                               pattern=pattern)

            logger.trace(f"features: {phase_features.get_attributes()}")
            predictions = load_and_predict(tier.bandwidth_model_path,
                                           pd.DataFrame(
                                               phase_features.get_attributes()),
                                           iops=True)
            logger.trace(f"predictions: {predictions.values.flatten()[0]}")
            return predictions.values.flatten()[0]

        if isinstance(tier.max_bandwidth, dict):
            return (tier.max_bandwidth[operation]['seq'] * pattern +
                    tier.max_bandwidth[operation]['rand'] * (1-pattern)) * cores * 1e6


        if isinstance(tier.max_bandwidth, (int, float)):
            return tier.max_bandwidth * cores
        else:
            # Try to cast max_bandwidth to a float
            try:
                max_bandwidth = float(tier.max_bandwidth) * cores
            except ValueError:
                # Handle the case where max_bandwidth cannot be cast to a float
                raise TypeError('max_bandwidth must be an int or float')

        return max_bandwidth * cores
        # elif isinstance(self.max_bandwidth, (int, float)):
        #     return self.max_bandwidth

        # else:
        #     tier_model = joblib.load(self.bandwidth_model_path)
        #     return self.max_bandwidth.predict([[cores, pattern]])[0]





class Tier:
    """
    A class to represent a tier in a cluster.

    Attributes:
        env (simpy.Environment): The simulation environment.
        name (str): The name of the tier.
        capacity (simpy.Container): The capacity of the tier.
        max_bandwidth (Union[Dict[str, Dict[int, float]], str, None]): The maximum bandwidth of the tier.
        bandwidth_model_path (Optional[str]): The path to the bandwidth model file.

    Raises:
        FileNotFoundError: If the bandwidth model file is not found.

    """

    def __init__(self, env, name, max_bandwidth=None, bandwidth_model_path=None, capacity=100e9):
        """
        Initializes a tier with the given parameters.

        Args:
            env (simpy.Environment): The simulation environment.
            name (str): The name of the tier.
            capacity (float, optional): The capacity of the tier in bytes. Defaults to 100e9.
            max_bandwidth (Union[Dict[str, Dict[int, float]], str, None], optional): The maximum bandwidth of the tier.
                Can be a dictionary containing bandwidth for different read/write operations and patterns,
                a string containing the path to the bandwidth model file, or None. Defaults to None.
            bandwidth_model_path (Optional[str], optional): The path to the bandwidth model file. Defaults to None.

        """
        self.env = env
        self.name = name
        self.capacity = simpy.Container(self.env, init=0, capacity=capacity)
        self.max_bandwidth = max_bandwidth
        self.bandwidth_model_path = bandwidth_model_path
        self.bandwidth = None
        self.bandwidth_concurrency = dict()

        # load the joblib model
        # if self.bandwidth_model_path:

        # if isinstance(max_bandwidth, str):
        #     if not bandwidth_model_path:
        #         raise FileNotFoundError("Bandwidth model file not found")
        #     with open(bandwidth_model_path) as f:
        #         bandwidth_model = pickle.load(f)
        #     self.max_bandwidth = bandwidth_model

    def get_max_bandwidth(self, cores=1, operation='read', pattern=1, new_data=None):
        """
        Returns the maximum bandwidth of the tier for the given parameters.

        Args:
            cores (int, optional): The number of cores. Defaults to 1.
            operation (str, optional): The read/write operation. Defaults to 'read'.
            pattern (int, optional): The access pattern (1 for sequential, 0 for random). Defaults to 1.

        Returns:
            float: The maximum bandwidth in bytes/sec.

        """
        if self.bandwidth_model_path:
            assert new_data is not None, "provide new data for bandwidth prediction"
            predictions = load_and_predict(self.bandwidth_model_path, new_data, iops=True)
            return predictions.values.flatten()

        if isinstance(self.max_bandwidth, dict):
            return (self.max_bandwidth[operation]['seq'] * pattern +
                    self.max_bandwidth[operation]['rand'] * (1-pattern)) * cores
        elif isinstance(self.max_bandwidth, (int, float)):
            return self.max_bandwidth

    def __str__(self):
        """
        Returns a string representation of the tier.

        Returns:
            str: A string representation of the tier.

        """
        description = "\n-------------------\n"
        description += (f"Tier:({type(self).__name__}) {self.name} with capacity = {convert_size(self.capacity.capacity)}\n")
        description += ("{:<12} {:<12} {:<12}".format('Operation', 'Pattern', 'Bandwidth MB/s')+"\n")
        if isinstance(self.max_bandwidth, dict):
            for op, inner_dict in self.max_bandwidth.items():
                for pattern, value in inner_dict.items():
                    description += ("{:<12} {:<12} {:<12}".format(op, pattern, value)+"\n")

        if isinstance(self.max_bandwidth, (int, float)):
            description += ("{:<12} {:<12} {:<12}".format("any", "any", float(self.max_bandwidth))+"\n")
        if self.bandwidth_model_path:
            description += f"Bandwidth model path: {self.bandwidth_model_path}\n"
        logger.trace(description)
        return description


class EphemeralTier(Tier):
    """Ephemeral tiers are tiers that are used for the duration of an application or a workflow. They are attached to a persistent Tier where data will resided at the end of the application or workflow.
    When the app should read data from a tier 1, and will make use of a transient tier, this data can be prefetched to the transient tier to be accessed later by the app from ephemeral tier.
    When the app should write data to a target tier 2, and will make use of a transient tier t, this data will be first written to tier t, and then when destaging policy is triggered destage data to tier 2.

    Args:
        persistent_tier(Tier): persistent tier attached to this transient/ephemeral tier where all data conveyed by the application will be found.
    """

    def __init__(self, env, name, persistent_tier, max_bandwidth, capacity=80e9):
        """Inits an EphemeralTier instance.

        Args:
            env (simpy.Environment): the simpy environment where all simulations happens.s
            name (string): unique name
            persistent_tier (Tier): a persistent tier the burst buffer is attached to. Will be automatically used for destaging purposes.
            max_bandwidth (simpy.Resource): bandwidth as limited number of slots that can be consumed.  Default capacity is up to 10 concurrent I/O sharing the maximum value of the bandwidth.
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
        super().__init__(env, name, max_bandwidth=max_bandwidth, capacity=capacity)

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
                logger.debug(f"{convert_size(eviction_volume)} evicted from tier {self.name}")
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


# class Tier:
#     """Model a tier storage service with a focus on a limited bandwidth resource as well as a limited capacity. In this model we expect a bandwidth value at its asymptotic state, so blocksize is still not a parameter. Only the asymptotic part of the throughput curve is considered. Other considered variables are read/write variables and sequential/random variables.
#     Output is a scalar value in MB/s.
#     Typically we access the bandwidth value as in dictionary: b['read']['seq'] = 200MB/s.
#     # TODO: extend this to a NN as function approximator to allow:
#         averaging over variables
#         interpolation when data entry is absent, i.e. b['seq'] gives a value

#     """

#     def __init__(self, env, name, capacity=100e9, max_bandwidth=None,  bandwidth_model_path=None):
#         """Inits a tier instance with some storage service specifications.

#         Args:
#             env (simpy.Environment): the simpy environment where all simulations happens.s
#             name (string): a name to make analytics readable and to assign a unique ID to a tier.
#             bandwidth (simpy.Resource): bandwidth as limited number of slots that can be consumed.  Default capacity is up to 10 concurrent I/O sharing the maximum value of the bandwidth.
#             capacity (simpy.Container, optional): storage capacity of the tier. Defaults to 100e9 (100GB).
#             max_bandwidth (dict): which contains operation and pattern dependant max bandwidths.
#             Example:
#                 ssd_bandwidth = {'read':  {'seq': 210, 'rand': 190},
#                          'write': {'seq': 100, 'rand': 100}}
#         """
#         self.env = env
#         self.name = name
#         self.capacity = simpy.Container(self.env, init=0, capacity=capacity)
#         self.max_bandwidth = max_bandwidth

#     def get_max_bandwidth(self, cores=1, operation='read', pattern=1):
#         if isinstance(self.max_bandwidth, dict):
#             return (self.max_bandwidth[operation]['seq'] * pattern +
#                     self.max_bandwidth[operation]['rand'] * (1-pattern)) * cores * 1e6
#         else:
#             return self.max_bandwidth.predict([[cores, pattern]])[0] * 1e6

#     def __str__(self):
#         """Prints cluster related information in a human readable fashion.

#         Returns:
#             string: properties and state of the cluster.
#         """
#         description = "\n-------------------\n"
#         description += (f"Tier: {self.name} with capacity = {convert_size(self.capacity.capacity)}\n")
#         description += ("{:<12} {:<12} {:<12}".format('Operation', 'Pattern', 'Bandwidth MB/s')+"\n")
#         for op, inner_dict in self.max_bandwidth.items():
#             for pattern, value in inner_dict.items():
#                 description += ("{:<12} {:<12} {:<12}".format(op, pattern, value)+"\n")
#         return description
