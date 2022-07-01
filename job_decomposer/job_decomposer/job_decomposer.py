#!/usr/bin/env python
"""
This module proposes a class decompose job timeseries and proposes a simple linear representation.
"""


__copyright__ = """
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

class JobDecomposer:
    def __init__(self, dataframe):
        """Init a JobDecomposer by accepting the related timeseries as

        Args:
            dataframe (pd.DataFrame): dataframe containing at least "timestamp", "bytesRead" and "bytesWritten" columns.
        """
        self.dataframe = dataframe
    
    def decompose(self, signal, algo, cost, pen):
        """Get breakpoints from algo, cost, pen.

        Args:
            algo (rpt Algo): algo for change point detection
            cost (_type_): cost function to perform optimization on the algo
            pen (_type_): penalty to add to cost function to find optimal number of breakpoints
        
        Returns:
            _type_: _description_
        """
        breakpoints = algo(custom_cost=cost).fit(signal).predict(pen)
        loss = cost.sum_of_costs(breakpoints)
        return breakpoints, loss
        
        
    def get_compute_vector(self, signal, breakpoints):
        """Get compute event list with the timeserie event from signal using found breakpoints

        Args:
            signal (_type_): _description_
            breakpoints (_type_): _description_
        """
        compute = [0]
        vector = [0]
        bandwidth = [0]
        closing_point = 0
        for i_brkpt, brkpt in enumerate(breakpoints[:-1]):
            if (i_brkpt % 2) == 0: # starting point
                starting_point = brkpt
                compute.append(compute[-1] + (starting_point - closing_point))
            if (i_brkpt % 2) != 0: # closing point
                closing_point = brkpt
                phase_volume = integrate.trapz(y=signal[starting_point: closing_point].flatten(), dx=5)
                vector.append(phase_volume)
                
        return compute, vector
                