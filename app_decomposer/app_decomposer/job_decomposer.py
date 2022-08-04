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

    def get_events_indexes(self, labels, signal):
        """Extract starting and ending indexes for each phase from the dataflow signal.

        Args:
            labels (list): list of labels associated with each signal point.
            signal (numpy.ndarray): the dataflow signal from which the events are extracted.

        Returns:
            tuple: two lists of indexes for starting and ending points for each detected phase within the signal.
        """

        label0 = get_lowest_cluster(labels, signal)
        # binarize labels to 0 (compute) and 1 (data phase)
        bin_labels = np.where(labels != label0, 1, 0)
        diff_array = np.diff(np.insert(bin_labels, 0, 0, axis=0))
        #print(f"diff array = {diff_array}")
        start_points = np.where(diff_array==1)[0].tolist()
        end_points = np.where(diff_array==-1)[0].tolist()
        # if there is an open phase at the end of the signal
        if len(end_points) == len(start_points) - 1:
            end_points.append(len(signal))
        assert len(start_points)==len(end_points)

        return start_points, end_points


    def get_signal_representation(self, timestamps, signal, labels):
        """Get compute event list with the timeserie event from signal using labels. The odd breakpoint opens a phase, an even one closes it. In between we sum the amount of data. Each couple of breakpoints are squeezed into a dirac representation having only one timestamp event.

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
        dx = np.diff(timestamps).tolist()[0]
        start_points, end_points = self.get_events_indexes(labels, signal)
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
            # print(f"start_index={start_index}, end_index={end_index}, duration={phase_duration}")
            compute.append(start_index - phase_duration)
            #phase_volume =
            phase_volume = np.sum(signal[start_index: end_index])
            data.append(phase_volume)
            bandwidth.append(phase_volume/((end_index - start_index)*dx))
            # should be min = 1

            phase_duration = end_index - start_index - 1
            # print(f"duration={phase_duration}")
            total_phase_durations += phase_duration

        #print(f"total_phase_durations={total_phase_durations}")

        # if phase end is not marked in signal, end it
        if end_points[-1] < len(signal):
            compute.append(len(signal) - 1 - total_phase_durations)
            data.append(0)
            bandwidth.append(0)

        return compute, data, bandwidth


    def get_job_representation(self):
        _, read_labels, _, write_labels = self.get_phases()
        read_events, read_volumes, read_bandwidths = self.get_signal_representation(self.timestamps, self.read_signal, read_labels)
        print(f"compute={read_events}, read={read_volumes}")
        write_events, write_volumes, write_bandwidths = self.get_signal_representation(self.timestamps, self.write_signal, write_labels)
        print(f"compute={write_events}, write={write_volumes}")

        events = np.unique(sorted(read_events + write_events)).tolist()
        reads = []
        writes = []
        bw = []

        for event in events:
            if event in read_events:
                reads.append(read_volumes[read_events.index(event)])
                #bw.append(read_bandwidths.index(event))
            else:
                reads.append(0)
            if event in write_events:
                writes.append(write_volumes[write_events.index(event)])
                #bw.append(write_events.index(event))
            else:
                writes.append(0)
        return events, reads, writes



















