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


def monitor(data, lst):
    state = "\n | Monitoring"
    for key, value in lst.items():
        state += "| " + key + ": " + str(value) + " "
    logger.debug(state)
    if isinstance(data, simpy.Store):
        data.put(lst)


class Cluster:
    def __init__(self, env, compute_nodes=1, cores_per_node=2, tiers=[], ephemeral_tier=None, data=None):
        self.compute_nodes = simpy.Resource(env, capacity=compute_nodes)
        self.compute_cores = simpy.Resource(env, capacity=cores_per_node*compute_nodes)
        self.data = data or None
        self.tiers = [tier for tier in tiers if isinstance(tier, Tier)]
        self.ephemeral_tier = ephemeral_tier if isinstance(ephemeral_tier, EphemeralTier) else None
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

    def get_max_bandwidth(self, tier, cores=1, operation='read', pattern=1):
        """Get the maximum bandwidth for a given tier, number of cores dedicated to the operation, a type of operation and an IO pattern."""
        if not isinstance(tier, Tier):
            tier = get_tier(self, tier)
        return (tier.max_bandwidth[operation]['seq'] * pattern +
                tier.max_bandwidth[operation]['rand'] * (1-pattern)) * cores*1e6

    def move_data(self, env, source_tier, target_tier, iophase, erase=False, data=None):
        """Move data from source_tier to target_tier.

        Args:
            env (Simpy.Environment): simulation environment.
            source_tier (Tier): the tier object where data is.
            target_tier (Tier): the tier where to move data to.
            iophase (IOPhase): the phase of the I/O operation.
        """
        self.env = env
        self.data = data if data else None
        # adjust source_tier level if necessary to be consistent
        if source_tier.capacity.level < iophase.volume:
            source_tier.capacity.put(iophase.volume - source_tier.capacity.level)
        target_free_space = (target_tier.capacity.capacity - target_tier.capacity.level)
        last_event = 0
        volume = iophase.volume
        while volume > 0:
            with source_tier.bandwidth.request() as source_req:
                with target_tier.bandwidth.request() as target_req:
                    # ensure both bw tokens are available
                    yield source_req & target_req
                    read_from_source_bandwidth = self.get_max_bandwidth(source_tier, operation='read', pattern=iophase.pattern)/source_tier.bandwidth.count
                    write_to_target_bandwidth = self.get_max_bandwidth(target_tier, operation='write', pattern=iophase.pattern)/target_tier.bandwidth.count
                    available_bandwidth = min(read_from_source_bandwidth, write_to_target_bandwidth)
                    time_to_complete = volume / available_bandwidth
                    next_event = self.env.peek()
                    step_duration = next_event - last_event if 0 < next_event - last_event < time_to_complete else time_to_complete
                    # define the duration event will take
                    step_event = self.env.timeout(step_duration)
                    t_start = self.env.now
                    yield step_event
                    t_end = self.env.now
                    volume -= step_duration * available_bandwidth
                    # update target_tier level
                    target_tier.capacity.put(step_duration * available_bandwidth)
                    if erase:
                        source_tier.capacity.get(step_duration * available_bandwidth)

                    monitoring_info = {"data_movement": {"source": source_tier.name, "target": target_tier.name},
                                       "t_start": t_start, "t_end": t_end,
                                       "bandwidth": available_bandwidth/1e6,
                                       "phase_duration": t_end-t_start,
                                       "volume": convert_size(step_duration * available_bandwidth),
                                       "tiers": [tier.name for tier in self.tiers],
                                       "tier_level": {tier.name: tier.capacity.level for tier in self.tiers}}
                    # when cluster include bb tier
                    if self.ephemeral_tier:
                        monitoring_info.update({self.ephemeral_tier.name + "_level":
                                                self.ephemeral_tier.capacity.level})

                    monitor(self.data, monitoring_info)
        return True

    # def bb_evict_data(self, env, bb_tier):
    #     """When flavor or datanode volume is full, start evicting at the speed of the lower tier.
    #     Args:
    #         env (simpy.Environment): env where all events are triggered.
    #     """
    #     # while True:
    #     #     if bb_tier.capacity.level >= 0.9*bb_tier.capacity.capacity:
    #     #         amount_to_destage = bb_tier.capacity.level - 0.9*bb_tier.capacity.capacity
    #     #         ret = yield env.process(self.run_stage(env, cluster, ))


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
        self.env = env
        self.name = name
        self.capacity = simpy.Container(self.env, init=0, capacity=capacity)
        self.max_bandwidth = bandwidth
        # modeling percent use of bandwidth
        self.bandwidth = simpy.Resource(self.env, capacity=10)
        self.bandwidth_concurrency = dict()
        # self.bandwidth = simpy.Container(env, init=100, capacity=100)
        # logger.info(self.__str__())

    def __str__(self):
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
        persistent_tier (Tier): persistent tier attached to this transient/ephemeral tier where all data conveyed by the application will be found.
    """

    def __init__(self, env, name, persistent_tier, bandwidth, capacity=80e9):
        # the non transient tier it is attached to
        self.persistent_tier = persistent_tier
        #self.cores = simpy.Resource(env, capacity=cores)
        super().__init__(env, name, bandwidth, capacity=capacity)


def bandwidth_share_model(n_threads):
    return np.sqrt(1 + n_threads)/np.sqrt(2)


def compute_share_model(n_cores):
    return np.sqrt(1 + n_cores)/np.sqrt(2)


def get_tier(cluster, tier_reference, use_bb=False):
    """Cluster has attributes called tiers and ephemeral_tier. The first one is a list and the second one is a single object attached to one of the (persistent) tiers.
    Reference to a tier could be either a string (search by name) or an integer (search by index in the list of tiers).
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
            tier (Tier): the tier object that was referenced."""
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
            # print("-----------oh-----")
            # print(type(tier_reference))
            # print(tier_reference.name)
            raise Exception(f"Tier {tier} not found")

        return tier
    if not use_bb:
        return find_tier(cluster, tier_reference)
    if use_bb:
        return cluster.ephemeral_tier
