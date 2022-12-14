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
from os.path import dirname, abspath
import numpy as np
from loguru import logger
from scipy import integrate
from app_decomposer.signal_decomposer import KmeansSignalDecomposer, get_lowest_cluster
from app_decomposer.api_connector import TimeSeries, MetaData, JobSearch
from app_decomposer.config_parser import Configuration
from app_decomposer.utils import convert_size
from app_decomposer import DEFAULT_CONFIGURATION, API_DICT_TS, IOI_SAMPLING_PERIOD, DATASET_SOURCE

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
    #print(f"labels={bin_labels}")
    diff_array = np.diff(np.insert(bin_labels, 0, 0, axis=0))
    #print(f"diff_array={diff_array}")
    start_points = np.where(diff_array==1)[0].tolist()
    end_points = np.where(diff_array==-1)[0].tolist()
    # if there is an open phase at the end of the signal
    if len(end_points) == len(start_points) - 1:
        end_points.append(len(signal))
    assert len(start_points)==len(end_points)
    #print(f"start points = {start_points}")
    #print(f"ending points = {end_points}")
    return start_points, end_points

def get_events_indexes_no_merge(labels, signal):
    """Extract starting and ending indexes for each phase from the dataflow signal. The indexes are computed from the labels. All labels different from label0 are considered as I/O phases.

    Args:
        labels (list): list of labels associated with each signal point.
        signal (numpy.ndarray): the dataflow signal from which the events are extracted.

    Returns:
        tuple: two lists of indexes for starting and ending points for each detected phase within the signal.
    """
    # print(labels)
    # print(type(labels))
    label0 = get_lowest_cluster(labels, signal)
    # adjust labels to 0 for compute phases and >1 for data phase
    ref_labels = np.where(labels != label0, np.array(labels) + 1, 0) # avoiding previous labeled 0
    #print(f"labels={ref_labels}")

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
            # print(f"sub_labels={sub_labels}")
            # print(f"diff_array={diff_array}")
            start_points = np.where(diff_array==1)[0].tolist()
            end_points = np.where(diff_array==-1)[0].tolist()
            # if there is an open phase at the end of the signal
            if len(end_points) == len(start_points) - 1:
                end_points.append(len(signal))
            assert len(start_points)==len(end_points)

            # print(f"start points = {start_points}")
            # print(f"end points = {end_points}")
            # adding sub phases to slicing
            for start_point, end_point in zip(start_points, end_points):
                #if (start_point, end_point) not in combinations:
                starting_points.append(start_point)
                ending_points.append(end_point)
    # sorting together
    if starting_points and ending_points:
        starting_points, ending_points = (list(t) for t in zip(*sorted(zip(starting_points, ending_points))))

    #print(f"starting points = {starting_points}")
    #print(f"ending points = {ending_points}")

    return starting_points, ending_points


def phases_to_representation(start_points, end_points, signal, dx=1):
    """Transform phase limits (starting points and ending points) into a representation in compute, data, and bandwidth lists.
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

def is_interval_in(starts, ends, i_start, i_end):
    """Check for a list of starting points and endings points (intervals) if they are contained in the master interval between i_start and i_end.

    Args:
        starts (list): list of starting points
        ends (list): list of ending points
        i_start (int): starting point of the comparison interval
        i_end (int): ending point of the comparison interval
    """
    result = []
    for starting_point, ending_point in zip(starts, ends):
        if i_start <= starting_point <= i_end and starting_point <= ending_point <= i_end:
            result.append(True)
        else:
            result.append(False)

    return result




def complex_to_representation(start_points, end_points, signal, dx=1):
    """Transform phase limits (starting points and ending points) into a representation in compute, data, and bandwidth lists.
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
    last_end = 0

    # if no breakpoints
    if not start_points and not end_points:
        compute, data, bandwidth = [0], [np.sum(signal)], [np.sum(signal)/len(signal)]
        return compute, data, bandwidth

    # if signal starts with compute
    if start_points[0] > 0:
        compute.append(0)
        data.append(0)
        bandwidth.append(0)

    # iterating over phases
    for start_index, end_index in zip(start_points, end_points):
        compute.append(start_index - total_phase_durations)
        # the IO between indexes will be reduced to dirac at start_index
        phase_volume = np.sum(signal[start_index: end_index])
        data.append(phase_volume)
        bandwidth.append(phase_volume/((end_index - start_index)*dx))
        phase_duration = end_index - start_index
        total_phase_durations += phase_duration
        last_end = end_index

    # print(f"total_phase_durations={total_phase_durations}")

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


def get_complex_representation(timestamps, signal, labels, merge_clusters=False):
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
    compute, data, bandwidth = complex_to_representation(start_points, end_points, signal, dx)

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
        # Initialize the IOI Connector Configuration
        self.config = Configuration(path=DEFAULT_CONFIGURATION)
        api_uri = f"{self.config.get_api_uri()}:{self.config.get_api_port()}"
        api_token = self.config.get_kc_token()
        self.timestamps, self.read_signal, self.write_signal = self.get_job_timeseries(api_uri, api_token)

    def get_job_timeseries(self, api_uri, api_token):
        """Method to extract read and write timeseries for a job instrumented in IOI.
        For the moment, data will be mocked by a csv file containing the timeseries.

        Returns:
            timestamps, read_signal, write_signal (numpy array): various arrays of the job timeseries.
        """
        job_connector = JobConnector(api_uri, api_token)
        return job_connector.get_data(self.job_id)

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
        #print(f"read_labels={read_labels}, write_labels={write_labels}")
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


class ComplexDecomposer:
    """This class takes separate read and write dataflow timeseries in order to extract separated phases for each type: compute, read and write phases."""
    def __init__(self, job_id=None, signal_decomposer=KmeansSignalDecomposer, v0_threshold=0.05,
                 config=Configuration(DEFAULT_CONFIGURATION)):
        """Initiates JobDecomposer class by fetching job related data.

        Args:
            job_id (int): slurm job id for which data will be retrieved.
            signal_decomposer(SignalDecomposer): class that decomposes each signal into phases.
        """
        self.job_id = job_id
        self.signal_decomposer = signal_decomposer
        self.v0_threshold = v0_threshold
        # Initialize the IOI Connector Configuration
        self.config = config #Configuration(path=DEFAULT_CONFIGURATION)
        api_uri = f"{self.config.get_api_uri()}:{self.config.get_api_port()}"
        api_token = self.config.get_kc_token()
        self.node_count = self.get_job_node_count(api_uri, api_token)
        self.timeseries = self.get_job_timeseries(api_uri, api_token)
        self.timestamps = self.timeseries["volume"]["timestamp"]
        self.read_signal = self.timeseries["volume"]["bytesRead"]
        self.write_signal = self.timeseries["volume"]["bytesWritten"]
        self.complex_signal = self.read_signal + 1j * self.write_signal
        self.norm_signal = np.abs(self.read_signal + 1j * self.write_signal)
        # NOTE: an alternative to complex norm
        # self.norm_signal = np.abs(self.read_signal) + np.abs(self.write_signal)
        assert self.timestamps.size > 0, "No data found for this job."


    def get_job_timeseries(self, api_uri, api_token):
        """Method to extract read and write timeseries for a job instrumented in IOI.
        For the moment, data will be mocked by a csv file containing the timeseries.

        Returns:
            timestamps, read_signal, write_signal (numpy array): various arrays of the job timeseries.
        """
        job_connector = JobConnector(api_uri, api_token)
        return job_connector.get_job_timeseries(self.job_id)

    def get_job_node_count(self, api_uri, api_token):
        """Retrieves node count data for a given job id.

        Returns:
            (int): data needed for job decomposition, node count and a dict containing all dataseries
        """
        job_connector = JobConnector(api_uri, api_token)
        object_id = job_connector.slurm_id_2_obj_id(self.job_id)
        return job_connector.get_node_count(object_id)

    def get_phases(self):
        """Get phases from each timeserie of the job but using mixed complex signal."""
        if len(self.norm_signal) == 1:
            return [], [0], [], [0], [], [0]
        else:
            read_decomposer = self.signal_decomposer(self.read_signal,
                                                     v0_threshold=self.v0_threshold)
            write_decomposer = self.signal_decomposer(self.write_signal,
                                                     v0_threshold=self.v0_threshold)
            norm_decomposer = self.signal_decomposer(self.norm_signal,
                                                     v0_threshold=self.v0_threshold)
            read_breakpoints, read_labels = read_decomposer.decompose()
            write_breakpoints, write_labels = write_decomposer.decompose()
            norm_breakpoints, norm_labels = norm_decomposer.decompose()

            return read_breakpoints, read_labels, write_breakpoints, write_labels, norm_breakpoints, norm_labels

    @staticmethod
    def get_dominant_pattern(pattern_freq, suffix="Read"):
        """Given a dict with pattern names and their respective frequencies. In case of equality, first key of the dict is returned by the max routine. Make sure "Uncl" pattern is the first entry of the pattern_freq dict.

        Example:
        {
            'accessUnclRead': 1,
            'accessRandRead': 2,
            'accessSeqRead': 3,
            'accessStrRead': 4
        }
        Should return 'Str' as it is the most frequent pattern.


        Args:
            suffix (str, optional): suffix to be removed from the pattern name. Defaults to "Read".
            pattern_freq (dict): _description_

        Returns:
            string: string name of the dominant pattern.
        """
        if not pattern_freq:
            return "Uncl"
        dominant_pattern = max(pattern_freq, key=pattern_freq.get) if pattern_freq else "Uncl"
        return dominant_pattern.split("access")[1].split(suffix)[0]


    def get_job_representation(self, merge_clusters=False):
        """Compute the job representation.

        Args:
            merge_clusters (bool, optional): Whether to merge points having belonging to one phase or not. Defaults to False.

        Returns:
            (dict): representation of the job as a set of numpy.ndarray.

        Example:
            {
                'events': [0, 1],
                'read_volumes': [0, 10],
                'read_bw': [0, 10.0],
                'write_volumes': [0, 40],
                'write_bw': [0, 40.0],
                'read_pattern': ['Uncl', 'Seq'],
                'write_pattern': ['Uncl', 'Str'],
                'read_operations': [0, 2],
                'write_operations': [0, 1]
            }

        """

        read_breakpoints, read_labels, write_breakpoints, write_labels, norm_breakpoints, norm_labels = self.get_phases()

        # print(f"norm bkps = {norm_breakpoints}")
        # print(f"norm labels = {norm_labels}")
        # print(f"read bkps = {read_breakpoints}")
        # print(f"read labels = {read_labels}")
        # print(f"write bkps = {write_breakpoints}")
        # print(f"write labels = {write_labels}")


        # we retain only compute array, volumes and bandwidths are computed individually
        # norm_compute, _, _ = get_complex_representation(self.timestamps, self.norm_signal,
        #                                                 norm_labels, merge_clusters=merge_clusters)
        if merge_clusters:
            norm_start_points, norm_end_points = get_events_indexes(norm_labels, self.norm_signal)
            read_start_points, read_end_points = get_events_indexes(read_labels, self.read_signal)
            write_start_points, write_end_points = get_events_indexes(write_labels, self.write_signal)
        else:
            norm_start_points, norm_end_points = get_events_indexes_no_merge(norm_labels, self.norm_signal)
            read_start_points, read_end_points = get_events_indexes_no_merge(read_labels, self.read_signal)
            write_start_points, write_end_points = get_events_indexes_no_merge(write_labels, self.write_signal)


        # integration step, only if size > 1
        dx = np.diff(self.timestamps.flatten()).tolist()[0] if self.timestamps.size > 1 else self.timestamps[0]
        dx = dx or 1
        compute = []
        read_bw = []
        read_volumes = []
        read_pattern = []
        read_operations = []

        write_bw = []
        write_volumes = []
        write_pattern = []
        write_operations = []

        output = {}
        phase_duration = 0
        excess_phase_durations = 0

        if not norm_start_points and not norm_end_points:
            if sum(self.norm_signal) > 0:
                # one I/O phase, assign extrem points and let do calculations
                norm_start_points = [0]
                norm_end_points = [len(self.norm_signal)]
            else:
                # pure compute phase
                output["node_count"] = self.node_count
                output["events"] = [0, len(self.norm_signal)-1]
                output["read_volumes"] = [0, 0]
                output["read_bw"] = [0, 0]
                output["write_volumes"] = [0, 0]
                output["write_bw"] = [0, 0]
                output["read_pattern"] = ["Uncl", "Uncl"]
                output["write_pattern"] = ["Uncl", "Uncl"]
                output["read_operations"] = [0, 0]
                output["write_operations"] = [0, 0]
                return output

        # adding 0 as default start point if not already existing
        if norm_start_points[0] > 0:
            old_start_point = norm_start_points[0]
            norm_start_points.insert(0, 0)
            norm_end_points.insert(0, old_start_point)
            # pairing this start point with first start point

        # appending end point with the last element of signal
        # if norm_end_points[-1] < len(self.norm_signal):
        #     norm_end_points.append(len(self.norm_signal))

        for i_start, i_end in zip(norm_start_points, norm_end_points):
            # feeding the compute array
            compute.append(i_start - excess_phase_durations)
            excess_phase_durations += i_end - i_start - 1

            ## feeding volumes and bandwidths
            # for a given "norm" phase
            read_volume = 0
            read_extent = 0
            access_read_pattern = {}
            read_op = 0

            # iterating subphases
            for starting_point, ending_point in zip(read_start_points, read_end_points):
                # iterate over read subphases
                #if i_start <= starting_point <= i_end and starting_point <= ending_point <= i_end:
                ending_point = min(ending_point, i_end)
                starting_point = min(max(starting_point, i_start), ending_point)
                # subphase within the norm phase
                read_volume += get_phase_volume(self.read_signal,
                                    start_index=starting_point,
                                    end_index=ending_point,
                                    dx=1)
                read_extent += ending_point - starting_point


                for access_key in ['accessUnclRead', 'accessRandRead', 'accessSeqRead', 'accessStrRead', ]:
                    # set default value to 0 if key does not exist
                    access_read_pattern.setdefault(access_key, 0)
                    if "accessPattern" in self.timeseries:
                        access_read_pattern[access_key] += get_phase_volume(self.timeseries["accessPattern"][access_key], start_index=starting_point, end_index=ending_point, dx=1)
                if "operationsCount" in self.timeseries:
                    read_op += get_phase_volume(self.timeseries["operationsCount"]["operationRead"], start_index=starting_point, end_index=ending_point, dx=1)

                assert ending_point >= starting_point
            read_volumes.append(read_volume)
            bw = read_volume/(read_extent) if read_extent else 0
            read_bw.append(bw)
            # Getting biggest access read
            read_pattern.append(ComplexDecomposer.get_dominant_pattern(access_read_pattern))
            read_operations.append(read_op)

            logger.info(f"Phase intervals {i_start}->{i_end} (dx={dx}) | read volume : {convert_size(read_volume)} | phase extent : {read_extent} | read bw : {convert_size(bw)}/s")

            write_volume = 0
            write_extent = 0
            access_write_pattern = {}
            write_op = 0
            for starting_point, ending_point in zip(write_start_points, write_end_points):
                # iterate over read subphases
                #if i_start <= starting_point <= i_end and starting_point <= ending_point <= i_end:
                ending_point = min(ending_point, i_end)
                starting_point = min(max(starting_point, i_start), ending_point)

                # subphase within the norm phase
                write_volume += get_phase_volume(self.write_signal,
                                    start_index=starting_point,
                                    end_index=ending_point,
                                    dx=1)
                write_extent += ending_point - starting_point

                for access_key in ['accessUnclWrite', 'accessRandWrite', 'accessSeqWrite', 'accessStrWrite']:
                    # set default value to 0 if key does not exist
                    access_write_pattern.setdefault(access_key, 0)
                    if "accessPattern" in self.timeseries:
                        access_write_pattern[access_key] += get_phase_volume(self.timeseries["accessPattern"][access_key], start_index=starting_point, end_index=ending_point, dx=1)

                if "operationsCount" in self.timeseries:
                    write_op += get_phase_volume(self.timeseries["operationsCount"]["operationWrite"], start_index=starting_point, end_index=ending_point, dx=1)


            write_volumes.append(write_volume)
            bw = write_volume/(write_extent) if write_extent else 0
            write_bw.append(bw)

            # Getting biggest access write
            write_pattern.append(ComplexDecomposer.get_dominant_pattern(access_write_pattern, suffix="Write"))
            write_operations.append(write_op)

            logger.info(f"Phase intervals {i_start}->{i_end} (dx={dx}) | write volume : {convert_size(write_volume)} | phase extent : {write_extent} | write bw : {convert_size(bw)}/s")

        if norm_end_points[-1] < len(self.norm_signal):
            compute.append(len(self.norm_signal) - 1*1 - excess_phase_durations)
            [output_list.append(0) for output_list in [read_volumes, write_volumes, read_bw, write_bw, read_operations, write_operations]]
            [output_list.append("Uncl") for output_list in [read_pattern, write_pattern]]


        # formatting output dict
        output["node_count"] = self.node_count
        output["events"] = compute
        output["read_volumes"] = read_volumes or [0]
        output["read_bw"] = read_bw or [0]
        output["write_volumes"] = write_volumes or [0]
        output["write_bw"] = write_bw or [0]
        output["read_pattern"] = read_pattern or ["Uncl"]
        output["write_pattern"] = write_pattern or ["Uncl"]
        output["read_operations"] = read_operations or [0]
        output["write_operations"] = write_operations or [0]

        return output

    @staticmethod
    def get_phases_features(representation, job_id=None, update_csv=False, dataset=None):
        """Builds from job representation an phases features dict to feed a performance model.
        Excludes phases having 0 volume (artefact of the decomposition).

        Args:
            representation (dict): job representation containing various information about the job.
            update
            dataset (path to csv file): if note False dump the representation in form of a csv file.
        Example:
            representation = {
                'events': [0, 1],
                'read_volumes': [0, 10],
                'read_bw': [0, 10.0],
                'write_volumes': [0, 40],
                'write_bw': [0, 40.0],
                'read_pattern': ['Uncl', 'Seq'],
                'write_pattern': ['Uncl', 'Str'],
                'read_operations': [0, 2],
                'write_operations': [0, 1]
                }

        Returns:
            list of dicts: features dict as showed in example below.

        Example:
            features = [{
                "volume": 1e9,
                "mode": "read" or "write",
                "IOpattern": "seq" or "str" or "rand" or "uncl",
                "IOsize": 4e3,
                "nodes": 1,
                "ioi_bw": 1e9,
                }, ...]
        """
        phases_features = []
        for i_phase, _ in enumerate(representation["events"]):
            features = {}
            # register job_id if known as well as nodes count
            features["job_id"] = job_id if job_id else "unknown"
            features["nodes"] = representation["node_count"]
            # job_id | read_volume | write_volume | read_io_pattern | write_io_pattern | read_io_size | write_io_size | nodes | ioi_bw
            features["read_volume"] = representation["read_volumes"][i_phase]
            features["write_volume"] = representation["write_volumes"][i_phase]
            features["read_io_pattern"] = representation["read_pattern"][i_phase].lower()
            features["write_io_pattern"] = representation["write_pattern"][i_phase].lower()

            features["read_io_size"] = representation["read_volumes"][i_phase] / representation["read_operations"][i_phase] if representation["read_operations"][i_phase] else 0

            features["write_io_size"] = representation["write_volumes"][i_phase] / representation["write_operations"][i_phase] if representation["write_operations"][i_phase] else 0

            # NOTE: this is an approximation were read_extent and write_extent are lost
            # phase duration is common to read and write ops within the same phase
            # volume_phase = volume_read + volume_write = read_bw*phase_duration + write_bw*phase_duration

            read_latency = representation["read_volumes"][i_phase]/representation["read_bw"][i_phase] if representation["read_bw"][i_phase] else 0
            write_latency =  representation["write_volumes"][i_phase]/representation["write_bw"][i_phase] if representation["write_bw"][i_phase] else 0
            sum_of_latencies = read_latency + write_latency
            sum_of_volumes = representation["read_volumes"][i_phase] + representation["write_volumes"][i_phase]
            # features["ioi_bw"] = (sum_of_volumes / sum_of_latencies) / IOI_SAMPLING_PERIOD if sum_of_latencies else 0
            features["ioi_bw"] = (representation["read_bw"][i_phase] +  representation["write_bw"][i_phase]) / IOI_SAMPLING_PERIOD

            phases_features.append(features)

        if update_csv:
            # Enable updating csv dataset file
            # dump the new features in the csv file
            pd.DataFrame(phases_features).to_csv(DATASET_SOURCE, mode='a', header=not os.path.exists(DATASET_SOURCE), index=False)
            # reset index
            #df = pd.read_csv(csv_path, index_col=False).reset_index(drop=True)
            #df.to_csv(csv_path, index=False)
            logger.info(f"Updating csv file with features: {DATASET_SOURCE}")

        if dataset:
            pd.DataFrame(phases_features).to_csv(dataset, index=False)
            logger.info(f"Dumping phases features in: {dataset}")

        return phases_features


class JobConnector:
    """Given a slurm job id, this class allows to retrieve volume timeseries and nodecount metadata to be used later by the job decomposer."""
    def __init__(self, api_uri, api_token):
        """Initializes the JobConnector class with the api token of the user that access the API.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
        """
        self.api_uri = api_uri
        self.api_token = api_token

    def slurm_id_2_obj_id(self, job_id):
        """Gets the object id of the database entry for job from the slurm job id reference.

        Args:
            job_id (int): slurm job id.

        Returns:
            str: object id that reference the db entry for job's information.
        """
        job_search = JobSearch(self.api_uri,
                               f"Bearer {self.api_token}",
                               job_filter={"jobid": {"contains": str(job_id)}})
        return job_search.job_objids_list[0]

    def get_node_count(self, object_id):
        """Get the compute node count that was used to run the job referenced by object_id.

        Args:
            object_id (str): object id that reference the db entry for job's information.

        Returns:
            int: number of compute nodes used to run the job.
        """
        metadata = MetaData(api_uri=self.api_uri,
                            api_token=f"Bearer {self.api_token}",
                            object_id=object_id)
        data = metadata.get_all_metadata()
        return data["nodeCount"]

    def get_job_volume(self, object_id):
        """Get volume timeseries and timestamps for a given object id.

        Args:
            object_id (str): object id that reference the db entry for job's information.

        Returns:
            tuple: three numpy ndarrays of data volumes.
        """
        time_series = TimeSeries(api_uri=self.api_uri,
                                 api_token=f"Bearer {self.api_token}",
                                 object_id=object_id,
                                 type_series='volume')
        data = time_series.get_data_by_label()
        return data["timestamp"], data["bytesRead"], data["bytesWritten"]

    def get_job_timeseries(self, job_id):
        """Get volume, operationsCount and accessPattern arrays from job's data.

        Args:
            object_id (str): object id that reference the db entry for job's information.

        Returns:
            job_timeseries (dict): with 'volume', 'operationsCount' and 'accesspattern' as keys.
            example:
            {
                'volume': {
                    'timestamp': array([0, 1], dtype=int64),
                    'bytesRead': array([0, 0], dtype=int64),
                    'bytesWritten': array([ 0, 42], dtype=int64)},
                'operationsCount': {
                    'timestamp': array([0, 1], dtype=int64),
                    'operationRead': array([0, 2], dtype=int64),
                    'operationWrite': array([0, 1], dtype=int64)},
                'accessPattern': {
                    'timestamp': array([0, 1], dtype=int64),
                    'accessRandRead': array([0, 12], dtype=int64),
                    'accessSeqRead': array([0, 12], dtype=int64),
                    'accessStrRead': array([0, 0], dtype=int64),
                    'accessUnclRead': array([40, 1], dtype=int64),
                    'accessRandWrite': array([0, 0], dtype=int64),
                    'accessSeqWrite': array([0, 0], dtype=int64),
                    'accessStrWrite': array([0, 0], dtype=int64),
                    'accessUnclWrite': array([0, 1], dtype=int64)
                }
            }
        """
        object_id = self.slurm_id_2_obj_id(job_id)
        return {ts:
            TimeSeries(api_uri=self.api_uri, api_token=f"Bearer {self.api_token}", object_id=object_id, type_series=ts).get_data_by_label() for ts in ["volume", "operationsCount", "accessPattern"]}

    def get_data(self, job_id):
        """Retrieves necessary data for JobDecomposer to extract phases and representation.

        Returns:
            tuple (int, dict): data needed for job decomposition, node count and a dict containing all dataseries
        """
        object_id = self.slurm_id_2_obj_id(job_id)
        node_count = self.get_node_count(object_id)
        timeseries = self.get_job_timeseries(object_id)
        return node_count, timeseries

