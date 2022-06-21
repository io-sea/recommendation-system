#!/usr/bin/env python
""" This module defines the rules used to detect for each timeframe of a job the corresponding
operating phase
"""
from __future__ import division, absolute_import, generators, print_function, unicode_literals,\
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2017 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import numpy as np


def start_end_detection(seq_io, op_io, seq_out):
    """ Compute the order 1 gradient of op_io and localize the max of operation in seq_io
    Note: op_io is a subset of seq_io (first/last 10% of timeframes)
    A start/end phase is composed of:
    - all the timeframes before/after the max of operation
    - the timeframes after/before the max of operation while the gradient of op_io
    is strictly negative

    Args:
        seq_io: boolean array of timeframe start or end
        op_io: array of metric used for the detection

    Returns:
        seq_out: boolean array of timeframe detected as start or end
    """
    i = 0
    diff_io = np.insert(np.diff(op_io), 0, 0)
    while (i <= (np.argmax(list(seq_io) or [0]))) | (diff_io[i] < 0):
        seq_out[i] = True
        i += 1
        if i >= len(diff_io):
            break
    return seq_out


def computeop_rw(data):
    """ Compute the number of read and write IO operations

    Args:
        jobid: id of the analyzed job
        dict_db: dictionnary of the loaded database

    Returns:
        tuple of array of number of read and write IO operations
    """
    r_dict = data[['read', 'write']]
    op_read = [np.sum(list(i.values())) for i in r_dict.read]
    op_write = [np.sum(list(i.values())) for i in r_dict.write]
    return (op_read, op_write)


def get_cut_start_end(data):
    """ Compute the timeframe threshold to detect start and end operating phases

    Args:
        jobid: id of the analyzed job
        dict_db: dictionnary of the loaded database

    Returns:
        threshold to detect start and end operating phases
    """
    return int(0.1*len(data.index))+1


def start_end_criteria(data, ptype):
    """ Criteria to detect start or end operating phases

    Args:
        jobid: id of the analyzed job
        dict_db: dictionnary of the loaded database
        ptype: flag string 'start' or 'end'

    Returns:
        boolean array of timeframe detected as start or end
    """
    cut_start_end = get_cut_start_end(data)
    phase = np.zeros((len(data.index)), dtype=bool)

    op_read, op_write = computeop_rw(data)
    op_io = np.sum(np.array([op_read, op_write]), axis=0)*\
                  np.array(data['bytesRead']+data['bytesWritten'])

    if ptype == 'start':
        op_io_start = op_io[0:cut_start_end]
        phase_detected = start_end_detection(op_io_start, op_io, phase)
    else:
        op_io_end = op_io[-cut_start_end:]
        rev_end = start_end_detection(list(reversed(op_io_end)), list(reversed(op_io)),
                                      list(reversed(phase)))
        phase_detected = list(reversed(rev_end))

    return np.array(phase_detected)


def undefined_criteria(data):
    """ Criteria to detect undefined operating phases (default phase of all timeframes)

    Args:
        jobid: id of the analyzed job
        dict_db: dictionnary of the loaded database

    Returns:
        boolean array of timeframe detected as undefined
    """
    return np.ones((data.shape[0])).astype(np.bool)


def ioinactive_criteria(data):
    """ Criteria to detect IO inactive operating phases

    Args:
        jobid: id of the analyzed job
        dict_db: dictionnary of the loaded database

    Returns:
        boolean array of timeframe detected as IO inactive
    """
    return np.array((data['ioActiveProcessCount'] < 1))


def ioactive_criteria(data):
    """ Criteria to detect IO active operating phases

    Args:
        jobid: id of the analyzed job
        dict_db: dictionnary of the loaded database

    Returns:
        boolean array of timeframe detected as IO active
    """
    return np.array((data['ioActiveProcessCount'] > 0))


def start_criteria(data):
    """ Criteria to detect start operating phases

    Args:
        jobid: id of the analyzed job
        dict_db: dictionnary of the loaded database

    Returns:
        boolean array of timeframe detected as start
    """
    return start_end_criteria(data, 'start')


def end_criteria(data):
    """ Criteria to detect end operating phases

    Args:
        jobid: id of the analyzed job
        dict_db: dictionnary of the loaded database

    Returns:
        boolean array of timeframe detected as end
    """
    return start_end_criteria(data, 'end')


def checkpoint_criteria(data):
    """ Criteria to detect checkpoint operating phases

    Args:
        jobid: id of the analyzed job
        dict_db: dictionnary of the loaded database

    Returns:
        boolean array of timeframe detected as checkpoint
    """
    rw_ratio = np.array((data['bytesRead']/data['bytesWritten']) < 0.2)

    _, op_write = computeop_rw(data)
    write_vol_op = np.array((op_write*data['bytesWritten']) > 1e9)

    cut_start_end = get_cut_start_end(data)
    start_end_mask = np.zeros((rw_ratio.size), dtype=np.bool)
    start_end_mask[cut_start_end:-cut_start_end] = True

    return rw_ratio & write_vol_op & start_end_mask


# models are defined as dictionnary of str keys and callable (function) values
MODEL = {'Undefined': undefined_criteria,
         'IOActive': ioactive_criteria,
         'IOInactive': ioinactive_criteria,
         'Start': start_criteria,
         'End': end_criteria,
         'Checkpoint': checkpoint_criteria,
         'requirements':  {'IODurationsGw': ['read', 'write'],
                           'FileIOSummaryGw': ['bytesRead', 'bytesWritten'],
                           'ProcessSummaryGw': ['ioActiveProcessCount']}}

CHCKPT_HIGHTLIGHTER = {'Undefined': undefined_criteria,
                       'Start': start_criteria,
                       'End': end_criteria,
                       'Checkpoint': checkpoint_criteria,
                       'requirements':  {'IODurationsGw': ['read', 'write'],
                                         'FileIOSummaryGw': ['bytesRead', 'bytesWritten']}}
