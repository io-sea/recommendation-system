#!/usr/bin/env python
"""
This module contains mainly utility functions that are used through other modules of the cluster simulator package.
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
import string
import random

# from phase import IOPhase


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


def compute_share_model(n_cores):
    """Description of parallelizing compute resources for an application. The gain factor is considered for a reference duration when using a single unit of computing.

    Args:
        n_cores (int): number of cores (computing unit) the application is distributed on.

    Returns:
        float: the speedup factor in comparison when using a single compute unit.
    """
    return np.sqrt(1 + n_cores)/np.sqrt(2)



class BandwidthResource(simpy.Resource):
    """Subclassing simpy Resource to introduce the ability to check_bandwidth when resource is requested or released."""

    def __init__(self, event_list, *args, **kwargs):
        """Init method using parent init method."""
        super().__init__(*args, **kwargs)
        self.env = args[0]
        self.event_list = event_list
        self.request_counter = {}
        self.release_counter = {}

    def request(self, *args, **kwargs):
        """On request method, check_bandwidth using parent request method."""
        logger.trace(f"bandwidth request at : {self.env.now}")
        self.request_counter[self.env.now] = self.request_counter.get(self.env.now, 0) + 1
        logger.trace(f"bandwidth request counter : {self.request_counter}")
        if self.request_counter[self.env.now] <= 2:
            self.check_bandwidth()
        ret = super().request(*args, **kwargs)
        # self.check_bandwidth()
        return ret

    def release(self, *args, **kwargs):
        """On release method, check_bandwidth using parent release method."""
        logger.trace(f"bandwidth release at : {self.env.now}")
        self.release_counter[self.env.now] = self.release_counter.get(self.env.now, 0) + 1
        logger.trace(f"bandwidth release counter : {self.release_counter}")
        if self.release_counter[self.env.now] <= 2:
            self.check_bandwidth()
        ret = super().release(*args, **kwargs)
        # self.check_bandwidth()
        return ret

    def check_bandwidth(self):
        """Checks running IO when bandwidth occupation changes. IOs should be interrupted on release or request of a bandwidth slot.
        """
        for io_event in self.event_list:
            # if not io_event.processed and io_event.triggered and io_event.is_alive:
            # if io_event.triggered and io_event.is_alive:
            if io_event.is_alive:

                # capture the IOs not finished, but triggered and alive
                logger.trace(f"({len(self.event_list)} events in queue |"
                             f"Event: {io_event.target}(trigg: {io_event.triggered} | live: {io_event.is_alive} | processed: {io_event.processed}) got interrupted at {self.env.now}")
                io_event.interrupt('updating bandwidth')


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


def name_app(number_of_letter=1, number_of_digits=1):
    """Give a random but reproducible string that should be enough to be unique for naming phases and applications.

    Returns:
        string: string concatenating uppercase letters and digits to be easily identifiable.
    """
    return ''.join(random.sample(string.ascii_uppercase, number_of_letter)) + ''.join([str(random.randint(0, 9)) for _ in range(number_of_digits)])


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

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1]
