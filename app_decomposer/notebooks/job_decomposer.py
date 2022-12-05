#!/usr/bin/env python
"""
This module proposes a class decompose job timeseries and proposes a simple linear representation.
"""


__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
""" = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import pandas as pd
import ruptures as rpt
import os, random


import warnings
warnings.filterwarnings("ignore")
class JobDecomposer:
    def __init__(self, dataframe):
        """Init a JobDecomposer by accepting the related timeseries as

        Args:
            dataframe (pd.DataFrame): dataframe containing at least "timestamp", "bytesRead" and "bytesWritten" columns.
        """
        self.dataframe = dataframe

    def get_breakpoints(self, signal, algo, cost, pen):
        """Get breakpoints from algo, cost, pen. Loss value indicates the quality of breakpoint detection.

        Args:
            algo (rpt Algo): algo for change point detection
            cost (_type_): cost function to perform optimization on the algo
            pen (_type_): penalty to add to cost function to find optimal number of breakpoints

        Returns:
            breakpoints (list): list of breakpoints indexes, last value of the list is the length of the signal.
            loss (float) : sum of costs for the chosen cost function.
        """
        breakpoints = algo(custom_cost=cost, min_size=1, jump=1).fit(signal).predict(pen)
        loss = cost.sum_of_costs(breakpoints)
        return breakpoints, loss


    def get_phases(self, x, signal, breakpoints):
        """Get compute event list with the timeserie event from signal using found breakpoints. The odd breakpoint opens a phase, an even one closes it. In between we sum the amount of data. Each couple of breakpoints are squeezed into a dirac representation having only one timestamp event.

        Args:
            signal (_type_): _description_
            breakpoints (_type_): indices of the detected changepoints.

        Returns:
            compute (list): list of timestamps events separated by compute phases.
            data (list) : associates an amount of data for each timestamped event. Could be related to write or read I/O phase.
            bandwidth : averaged bandwidth as a constant value through the phase.
        """
        compute = [0]
        data = []
        bandwidth = [0]
        closing_point = 0
        dx = np.diff(x).tolist()[0]
        for i_brkpt, brkpt in enumerate(breakpoints[:-1]): # last brkpoint is length of signal
            if (i_brkpt % 2) == 0: # starting phase
                starting_point = brkpt
                compute.append(compute[-1] + (starting_point - closing_point))
                phase_volume = integrate.trapz(y=signal[starting_point: closing_point], dx=dx)
                data.append(phase_volume)

            if (i_brkpt % 2) != 0: # closing phase
                closing_point = brkpt
                phase_volume = integrate.trapz(y=signal[starting_point: closing_point], dx=dx)
                phase_length = closing_point - starting_point
                data.append(phase_volume)
                bandwidth.append(phase_volume/phase_length)


        return compute, data, bandwidth
