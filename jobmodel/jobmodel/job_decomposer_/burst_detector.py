#!/usr/bin/env python
"""
This module creates class and implements methods necessary for implementing
Kleinberg's burst detection analysis on batched data.
https://www.cs.cornell.edu/home/kleinber/bhs.pdf
https://github.com/nmarinsek/burst_detection/blob/master/burst_detection/__init__.py
https://github.com/nmarinsek/burst_detection/blob/master/burst_detection_validation.ipynb
"""
from __future__ import division, absolute_import, generators, print_function, unicode_literals, \
    with_statement, nested_scopes  # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2019 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import pandas as pd
import numpy as np
from sympy import binomial


class BurstDectector:
    def __init__(self, signal):
        self.signal = signal
        self.d = np.ones(shape=self.signal.shape) * np.max(self.signal)
        n = len(signal)
        # [q, _, _, p] = bd.burst_detection(signal, d, len(signal), 1.5, 2, smooth_win=0)

    @staticmethod
    def tau(i1, i2, gamma, n):
        """
        Define the transition cost tau: cost of switching states
        there's a cost to move up states, no cost to move down
        based on definition on pg. 8
        inputs
        i1: current state
        i2: next state
        gam: gamma, penalty for moving up a state
        n: number of timepoints

        """
        if i1 >= i2:
            return 0
        else:
            return (i2 - i1) * gamma * np.log(n)

    @staticmethod
    def fit(d, r, p):
        """ define the fit cost: goodness of fit to the expected outputs of each state
        based on equation on bottom of pg. 14
        d: number of events in each time period (1xn)
        r: number of target events in each time period (1xn)
        p: expected proportions of each state (1xk)
        """
        return -np.log(np.float(binomial(d, r)) * (p ** r) * (1 - p) ** (d - r))

    def burst_detection(self, r, d, n, s, gamma):
        """
        # define the burst detection function for a two-state automaton
    # inputs:
    #   r: number of target events in each time period (1xn)
    #   d: number of events in each time period (1xn)
    #   n: number of timepoints
    #   s: multiplicative distance between states
    #   gamma: difficulty to move up a state
    #   smooth_win: width of smoothing window (use odd numbers)
    # output:
    #   q: optimal state sequence (1xn)
        """
        k = 2  # two states

        # calculate the expected proportions for states 0 and 1
        self.p = {}
        # overall proportion of events, baseline state
        self.p[0] = np.nansum(r) / float(np.nansum(d))
        self.p[1] = self.p[0] * s  # proportion of events during active state
        if self.p[1] > 1:  # p1 can't be bigger than 1
            self.p[1] = 0.99999

        # initialize matrices to hold the costs and optimal state sequence
        cost = np.full([n, k], np.nan)
        self.q = np.full([n, 1], np.nan)

        # use the Viterbi algorithm to find the optimal state sequence
        for t in range(0, n):
            # calculate the cost to transition to each state
            for j in range(k):
                # for the first timepoint, calculate the fit cost only
                if t == 0:
                    cost[t, j] = self.fit(d[t], r[t], p[j])
                # for all other timepoints, calculate the fit and transition cost
                else:
                    cost[t, j] = self.tau(self.q[t - 1], j, gamma, n) + self.fit(d[t], r[t], p[j])
            # add the state with the minimum cost to the optimal state sequence
            self.q[t] = np.where(cost[t, :] == min(cost[t, :]))

    def enumerate_bursts(self, q, label):
        """
        # define a function to enumerate the bursts
        # input:
        #   q: optimal state sequence
        # output:
        #   bursts: dataframe with beginning and end of each burst
        :param label:
        :return:
        """
        bursts = pd.DataFrame(columns=['label', 'begin', 'end', 'weight'])
        b = 0
        burst = False
        for t in range(1, len(self.q)):

            if (burst is False) & (self.q[t] > self.q[t - 1]):
                bursts.loc[b, 'begin'] = t
                burst = True

            if (burst is True) & (self.q[t] < self.q[t - 1]):
                bursts.loc[b, 'end'] = t
                burst = False
                b = b + 1

        # if the burst is still going, set end to last timepoint
        if burst is True:
            bursts.loc[b, 'end'] = t

        bursts.loc[:, 'label'] = label
        return bursts

    def burst_weights(self, bursts, r, d, p):
        """

    # define a function that finds the weights associated with each burst
    # find the difference in the cost functions for p0 and p1 in each burst
    # inputs:
    #   bursts: dataframe containing the beginning and end of each burst
    #   r: number of target events in each time period
    #   d: number of events in each time period
    #   p: expected proportion for each state
    # output:
    #   bursts: dataframe containing the weights of each burst, in order
        """
        # loop through bursts
        for b in range(len(bursts)):
            cost_diff_sum = 0
            for t in range(bursts.loc[b, 'begin'], bursts.loc[b, 'end']):
                cost_diff_sum = cost_diff_sum + \
                                (self.fit(d[t], r[t], p[0]) - self.fit(d[t], r[t], p[1]))
            bursts.loc[b, 'weight'] = cost_diff_sum
        return bursts.sort_values(by='weight', ascending=False)
