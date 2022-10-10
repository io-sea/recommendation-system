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
from loguru import logger
from scipy import integrate
from app_decomposer.signal_decomposer import KmeansSignalDecomposer, get_lowest_cluster
from app_decomposer.api_connector import TimeSeries, MetaData, JobSearch
from app_decomposer.config_parser import Configuration
from app_decomposer import DEFAULT_CONFIGURATION

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
    # print(labels)
    # print(type(labels))
    label0 = get_lowest_cluster(labels, signal)
    # adjust labels to 0 for compute phases and >1 for data phase
    ref_labels = np.where(labels != label0, np.array(labels) + 1, 0) # avoiding previous labeled 0
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

    print(f"starting points = {starting_points}")
    print(f"ending points = {ending_points}")

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


class ComplexDecomposer:
    """This class takes separate read and write dataflow timeseries in order to extract separated phases for each type: compute, read and write phases."""
    def __init__(self, job_id=None, signal_decomposer=KmeansSignalDecomposer, v0_threshold=0.05):
        """Initiates JobDecomposer class by fetching job related data.

        Args:
            job_id (int): slurm job id for which data will be retrieved.
            signal_decomposer(SignalDecomposer): class that decomposes each signal into phases.
        """
        self.job_id = job_id
        self.signal_decomposer = signal_decomposer
        self.v0_threshold = v0_threshold
        # Initialize the IOI Connector Configuration
        self.config = Configuration(path=DEFAULT_CONFIGURATION)
        api_uri = f"{self.config.get_api_uri()}:{self.config.get_api_port()}"
        api_token = self.config.get_kc_token()
        self.timestamps, self.read_signal, self.write_signal = self.get_job_timeseries(api_uri, api_token)
        self.complex_signal = self.read_signal + 1j * self.write_signal
        self.norm_signal = np.abs(self.read_signal + 1j * self.write_signal)
        #self.norm_signal = np.abs(self.read_signal) + np.abs(self.write_signal)
        self.read_rec = None
        self.write_rec = None
        self.norm_rec = None

    def get_job_timeseries(self, api_uri, api_token):
        """Method to extract read and write timeseries for a job instrumented in IOI.
        For the moment, data will be mocked by a csv file containing the timeseries.

        Returns:
            timestamps, read_signal, write_signal (numpy array): various arrays of the job timeseries.
        """
        job_connector = JobConnector(api_uri, api_token)
        return job_connector.get_data(self.job_id)

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
            self.read_rec = read_decomposer.reconstruct(read_breakpoints)
            write_breakpoints, write_labels = write_decomposer.decompose()
            self.write_rec = write_decomposer.reconstruct(write_breakpoints)
            norm_breakpoints, norm_labels = norm_decomposer.decompose()
            self.norm_rec = norm_decomposer.reconstruct(norm_breakpoints)

            return read_breakpoints, read_labels, write_breakpoints, write_labels, norm_breakpoints, norm_labels

    def get_job_representation(self, merge_clusters=False):

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
        write_bw = []
        write_volumes = []

        phase_duration = 0
        excess_phase_durations = 0

        # # if no breakpoints, no I/O phases
        # if not read_breakpoints and not write_breakpoints:
        #     if len(self.norm_signal) == 1:
        #         return [0], [0], [0], [0], [0]
        #     else:
        #         return [0, len(self.norm_signal)-1], [0, 0], [0, 0], [0, 0], [0, 0]


        # if no breakpoints, no I/O phases
        # if not norm_start_points and not norm_end_points:
        #     if len(self.norm_signal) == 1:
        #         return [0], [0], [0], [0], [0]
        #     else:
        #         return [0, len(self.norm_signal)-1], [0, 0], [0, 0], [0, 0], [0, 0]

        # if no phases
        if not norm_start_points and not norm_end_points:
            if sum(self.norm_signal) > 0:
                # one I/O phase, assign extrem points and let do calculations
                norm_start_points = [0]
                norm_end_points = [len(self.norm_signal)]
            else:
                # pure compute phase
                return [0, len(self.norm_signal)-1], [0, 0], [0, 0], [0, 0], [0, 0]

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
                assert ending_point >= starting_point
            read_volumes.append(read_volume)
            bw = read_volume/(read_extent + 1) if read_extent else 0
            read_bw.append(bw)

            logger.info(f"Phase intervals {i_start}->{i_end} (dx={dx}) | read volume : {read_volume} | phase extent : {read_extent} | read bw : {bw}")

            write_volume = 0
            write_extent = 0
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
            write_volumes.append(write_volume)
            bw = write_volume/(write_extent + 1) if write_extent else 0
            write_bw.append(bw)

            logger.info(f"Phase intervals {i_start}->{i_end} (dx={dx}) | write volume : {write_volume} | phase extent : {write_extent} | write bw : {bw}")

        if norm_end_points[-1] < len(self.norm_signal):
            compute.append(len(self.norm_signal) - 1 - excess_phase_durations)
            [output_list.append(0) for output_list in [read_volumes, write_volumes, read_bw, write_bw]]

        read_volumes = read_volumes or [0]
        read_bw = read_bw or [0]
        write_volumes = write_volumes or [0]
        write_bw = write_bw or [0]

        return compute, read_volumes, read_bw, write_volumes, write_bw


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

    def get_job_data(self, object_id):
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

    def get_data(self, job_id):
        """Retrieves necessary data for JobDecomposer to extract phases and representation.

        Returns:
            tuple: data needed for job decomposition.
        """
        object_id = self.slurm_id_2_obj_id(job_id)
        node_count = self.get_node_count(object_id)
        return self.get_job_data(object_id)






