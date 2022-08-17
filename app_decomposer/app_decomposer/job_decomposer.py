#!/usr/bin/env python
"""
This module proposes classes and methods to gather various signals from instrumentation to combine I/O activites into phases distinct from compute activities.
"""


__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import os
import pandas as pd
import random
import numpy as np
from scipy import integrate
from app_decomposer.signal_decomposer import KmeansSignalDecomposer, get_lowest_cluster


def get_phase_volume(signal, method="sum", start_index=0, end_index=-1, dx=1):
    """Method allowing many functions to compute the total volume of data for a given phase boundary in a signal array.
    TODO : insert in code
    Args:
        signal (numpy.ndarray): signal for which the volume is computed.
        method (str, optional): description of the method. Defaults to "sum".
        start_index (int, optional): phase start index. Defaults to 0.
        end_index (int, optional): phase end index. Defaults to -1.
        dx (int, optional): sampling period of the signal. Defaults to 1.

    Returns:
        float: the total volume of the phase computed from the signal using the chosen method.
    """
    end_index = len(signal) if end_index == -1 or end_index >= len(signal) else end_index
    if method == "sum":
        return np.sum(signal[start_index: end_index])
    elif method == "trapz":
        return integrate.trapz(y=signal[start_index: end_index], dx=dx)
    elif method == "simps":
        return integrate.simps(y=signal[start_index: end_index], dx=dx)

#

def get_events_indexes(labels, signal):
    """Extract starting and ending indexes for each phase from the dataflow signal. The indexes are computed from the labels. All labels different from label0 are merged into label1.

    Args:
        labels (list): list of labels associated with each signal point.
        signal (numpy.ndarray): the dataflow signal from which the events are extracted.

    Returns:
        tuple: two lists of indexes for starting and ending points for each detected phase within the signal.
    """

    label0 = get_lowest_cluster(labels, signal)
    # binarize labels to 0 (compute) and 1 (data phase)
    bin_labels = np.where(labels != label0, 1, 0)
    print(f"labels={bin_labels}")
    diff_array = np.diff(np.insert(bin_labels, 0, 0, axis=0))
    print(f"diff_array={diff_array}")
    start_points = np.where(diff_array==1)[0].tolist()
    end_points = np.where(diff_array==-1)[0].tolist()
    # if there is an open phase at the end of the signal
    if len(end_points) == len(start_points) - 1:
        end_points.append(len(signal))
    assert len(start_points)==len(end_points)
    print(f"start points = {start_points}")
    print(f"ending points = {end_points}")
    return start_points, end_points

def get_events_indexes_no_merge(labels, signal):
    """Extract starting and ending indexes for each phase from the dataflow signal. The indexes are computed from the labels. All labels different from label0 are considered as I/O phases.

    Args:
        labels (list): list of labels associated with each signal point.
        signal (numpy.ndarray): the dataflow signal from which the events are extracted.

    Returns:
        tuple: two lists of indexes for starting and ending points for each detected phase within the signal.
    """
    label0 = get_lowest_cluster(labels, signal)
    # adjust labels to 0 for compute phases and >1 for data phase
    ref_labels = np.where(labels != label0, labels + 1, 0) # avoiding previous labeled 0
    print(f"labels={ref_labels}")

    starting_points = []
    ending_points = []

    diff_array = np.diff(np.insert(ref_labels, 0, 0, axis=0))
    for label_value in np.unique(ref_labels):
        # combinations = []
        # for start, end in zip(starting_points, ending_points):
        #     combinations.append((start, end))
        # print(combinations)
        if label_value != label0:

            sub_labels = np.where(ref_labels == label_value, 1, 0)
            diff_array = np.diff(np.insert(sub_labels, 0, 0, axis=0))
            print(f"sub_labels={sub_labels}")
            print(f"diff_array={diff_array}")
            start_points = np.where(diff_array==1)[0].tolist()
            end_points = np.where(diff_array==-1)[0].tolist()
            # if there is an open phase at the end of the signal
            if len(end_points) == len(start_points) - 1:
                end_points.append(len(signal))
            assert len(start_points)==len(end_points)

            print(f"start points = {start_points}")
            print(f"end points = {end_points}")
            # adding sub phases to slicing
            for start_point, end_point in zip(start_points, end_points):
                #if (start_point, end_point) not in combinations:
                starting_points.append(start_point)
                ending_points.append(end_point)
    # sorting together
    starting_points, ending_points = (list(t) for t in zip(*sorted(zip(starting_points, ending_points))))

    print(f"starting points = {starting_points}")
    print(f"ending points = {ending_points}")

    return starting_points, ending_points


def phases_to_representation(start_points, end_points, signal, dx=1):
    """Transform phase limits (starting points  and ending points) into a representation in compute, data, and bandwidth lists.
    Args:
        timestamps (numpy.array): timestamp array where for each value a measure was done.
        signal (numpy.ndarray): _description_
        breakpoints (list): indices of the detected changepoints.

    Returns:
        compute (list): list of timestamps events separated by compute phases.
        data (list) : associates an amount of data for each timestamped event. Could be related to write or read I/O phase.
        bandwidth : averaged bandwidth as a constant value through the phase.
    """
    compute = []
    data = []
    bandwidth = []
    phase_duration = 0
    total_phase_durations = 0
    # if signal starts with compute
    if start_points[0] > 0:
        compute.append(0)
        data.append(0)
        bandwidth.append(0)

    # iterating over phases
    for start_index, end_index in zip(start_points, end_points):
        # the IO between indexes will be reduced to dirac at start_index
        compute.append(start_index - 0*phase_duration - total_phase_durations)
        phase_volume = np.sum(signal[start_index: end_index])
        data.append(phase_volume)
        bandwidth.append(phase_volume/((end_index - start_index)*dx))
        phase_duration = end_index - start_index - 1
        total_phase_durations += phase_duration
    #print(f"total_phase_durations={total_phase_durations}")

    # if phase end is not marked in signal, end it
    if end_points[-1] < len(signal):
        compute.append(len(signal) - 1 - total_phase_durations)
        data.append(0)
        bandwidth.append(0)

    return compute, data, bandwidth

def get_signal_representation(timestamps, signal, labels, merge_clusters=False):
    """Get compute event list with the timeserie event from signal using labels. The odd breakpoint opens a phase, an even one closes it. In between we sum the amount of data. Each couple of breakpoints are squeezed into a dirac representation having only one timestamp event.

    Args:
        timestamps (numpy.array): timestamp array where for each value a measure was done.
        signal (numpy.ndarray): _description_
        labels (list): indices to distinguish each ordinate value from the signal.
        merge_cluster (bool): merge all labels different from label0 into label 1 if true, else keep distinguishing between ordinates value. Defaults to False

    Returns:
        compute (list): list of timestamps events separated by compute phases.
        data (list) : associates an amount of data for each timestamped event. Could be related to write or read I/O phase.
        bandwidth (list) : averaged bandwidth as a constant value through the phase.
    """

    dx = np.diff(timestamps.flatten()).tolist()[0]
    if merge_clusters:
        start_points, end_points = get_events_indexes(labels, signal)
    else:
        start_points, end_points = get_events_indexes_no_merge(labels, signal)
    compute, data, bandwidth = phases_to_representation(start_points, end_points, signal, dx)

    return compute, data, bandwidth



class JobDecomposer:
    """This class takes separate read and write dataflow timeseries in order to extract separated phases for each type: compute, read and write phases."""
    def __init__(self, job_id=None, signal_decomposer=KmeansSignalDecomposer):
        """Initiates JobDecomposer class by fetching job related data.

        Args:
            job_id (int): slurm job id for which data will be retrieved.
            signal_decomposer(SignalDecomposer): class that decomposes each signal into phases.
        """
        self.job_id = job_id
        self.signal_decomposer = signal_decomposer
        self.timestamps, self.read_signal, self.write_signal = self.get_job_timeseries()

    def get_job_timeseries(self):
        """Method to extract read and write timeseries for a job instrumented in IOI.
        TODO: connect this method directly to the IOI database.
        For the moment, data will be mocked by a csv file containing the timeseries.

        Returns:
            timestamps, read_signal, write_signal (numpy array): various arrays of the job timeseries.
        """
        return 0, 0, 0

    def get_phases(self):
        """Get phases from each timeserie of the job."""
        read_decomposer = self.signal_decomposer(self.read_signal)
        write_decomposer = self.signal_decomposer(self.write_signal)
        read_breakpoints, read_labels = read_decomposer.decompose()
        write_breakpoints, write_labels = write_decomposer.decompose()
        return read_breakpoints, read_labels, write_breakpoints, write_labels


    def get_job_representation(self, merge_clusters=False):
        """Uses jobs data signals to produce a formal representation as an events list with either compute or data phases. Other arrays are produced to indicate the amount of data for each phase, the recorder bandwidth and (TODO) the pattern of the phases.

        Returns:
            events (list): list of indexes/timestamps for each phase. If it is marked by non zero value in the read_volumes or write_volumes lists, then it is read/write or both I/O phase. If not, a compute phase until the next event.
            read_volumes (list): list of amount of data for each read phase.
            write_volumes (list): list of amount of data for each write phase.
        """
        _, read_labels, _, write_labels = self.get_phases()
        print(f"read_labels={read_labels}, write_labels={write_labels}")
        dx = np.diff(self.timestamps.flatten()).tolist()[0]
        read_events, read_volumes_, read_bandwidths = get_signal_representation(self.timestamps, self.read_signal, read_labels, merge_clusters=merge_clusters)
        print(f"compute={read_events}, read={read_volumes_}")
        write_events, write_volumes_, write_bandwidths = get_signal_representation(self.timestamps, self.write_signal, write_labels, merge_clusters=merge_clusters)
        print(f"compute={write_events}, write={write_volumes_}")
        events = np.unique(sorted(read_events + write_events)).tolist()
        job_compute = []
        read_volumes = []
        write_volumes = []
        read_bw = []
        write_bw = []
        for i_event, event in enumerate(events):
            print(f"event idx = {i_event} = ===================")
            print(f"event list = {events} = ===================")
            print(f"compute={events}, read={read_volumes}, writes={write_volumes}")

            # check if event is read
            if event in read_events:
                read_volumes.append(read_volumes_[read_events.index(event)])
                read_bw.append(read_bandwidths[read_events.index(event)])
            else:
                read_volumes.append(0)
                read_bw.append(0)

            # check if event is read
            if event in write_events:
                write_volumes.append(write_volumes_[write_events.index(event)])
                write_bw.append(write_bandwidths[write_events.index(event)])
            else:
                write_volumes.append(0)
                write_bw.append(0)
            job_compute.append(event)
            print(f"compute={job_compute}, read={read_volumes}, writes={write_volumes}")

        return job_compute, read_volumes, write_volumes, read_bw, write_bw


# def combine_representation(representation1, representation2):
#     """Considers I/O phase exhibiting various clusters levels (levels of bandwidths) to be segmented accoringgly.

#     Args:
#         representation1 (tuple): contains list of event timestamps, their relative volumes and bandwidths.
#         representation2 (tuple): contains list of event timestamps, their relative volumes and bandwidths.

#     Returns:
#         _type_: _description_
#     """
#     events1, data1, bw1 = representation1
#     events2, data2, bw2 = representation2

#     events = np.unique(sorted(events1 + events2)).tolist()
#     data = [0]*len(events)
#     bw = [0]*len(events)

#     for idx, event in enumerate(events):
#         if event in events1:
#             data[idx] = data1[events1.index(event)]
#             bw[idx] = bw1[events1.index(event)]
#         # else:
#         #     data.append(0)
#         #     bw.append(0)
#         if event in events2:
#             data[idx] = data2[events2.index(event)]
#             bw[idx] = bw2[events2.index(event)]
#         # else:
#         #     data.append(0)
#         #     bw.append(0)
#     return events, data, bw










