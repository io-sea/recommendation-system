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
