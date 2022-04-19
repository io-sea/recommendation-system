import simpy
from loguru import logger
import numpy as np
import math
from monitor import MonitorResource


def convert_size(size_bytes):
    # later replace 1000 by 1024
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1000)))
    p = math.pow(1000, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def get_tier(tier, cluster):
    if isinstance(tier, int):
        tier = cluster.tiers[tier]
    if isinstance(tier, str):
        for cluster_tier in cluster.tiers:
            if tier == cluster_tier.name:
                tier = cluster_tier
    return tier


class Cluster:
    def __init__(self, env, compute_nodes=1, cores_per_node=2, tiers=[]):
        # self.compute_nodes = simpy.Container(env, capacity=compute_nodes)
        # self.compute_cores = simpy.Container(env, capacity=cores_per_node*compute_nodes)
        self.compute_nodes = simpy.Resource(env, capacity=compute_nodes)
        self.compute_cores = simpy.Resource(env, capacity=cores_per_node*compute_nodes)
        self.tiers = []
        for tier in tiers:
            if isinstance(tier, Tier):
                self.tiers.append(tier)
        # logger.info(self.__str__())

    def __str__(self):
        description = "====================\n"
        description += (f"Cluster with {self.compute_nodes.capacity} compute nodes \n")
        description += f"Each having {self.compute_cores.capacity} cores in total \n"
        if self.tiers:
            for tier in self.tiers:
                description += tier.__str__()
        return description
        # self.storage_capacity = simpy.Container(env, init=0, capacity=storage_capacity)
        # self.storage_speed = simpy.Container(env, init=storage_speed, capacity=storage_speed)


class Tier:
    """
    In this model we expect a bandwidth value at its asymptotic state.
    Only the maximum is considered.
    Other considered variables are :
        read/write variables
        sequential/random variables
    Output is a scalar value in MB/s.
    Typically we access the bandwidth value as in dictionary: b['read']['seq'] = 200MB/s.
    TODO : extend this to a NN as function approximator to allow:
        averaging over variables
        interpolation when data entry is absent, i.e. b['seq'] gives a value
    """

    def __init__(self, env, name, bandwidth, capacity=100e9):
        self.name = name
        self.capacity = simpy.Container(env, init=0, capacity=capacity)
        self.max_bandwidth = bandwidth
        # modeling percent use of bandwidth
        self.bandwidth = simpy.Resource(env, capacity=10)
        #self.bandwidth = simpy.Container(env, init=100, capacity=100)
        # logger.info(self.__str__())

    def __str__(self):
        description = "-------------------\n"
        description += (f"Tier: {self.name} with capacity = {convert_size(self.capacity.capacity)}\n")
        description += ("{:<12} {:<12} {:<12}".format('Operation', 'Pattern', 'Bandwidth MB/s')+"\n")
        for op, inner_dict in self.max_bandwidth.items():
            for pattern, value in inner_dict.items():
                description += ("{:<12} {:<12} {:<12}".format(op, pattern, value)+"\n")
        return description


def bandwidth_share_model(n_threads):
    return np.sqrt(1 + n_threads)/np.sqrt(2)


def compute_share_model(n_cores):
    return np.sqrt(1 + n_cores)/np.sqrt(2)
