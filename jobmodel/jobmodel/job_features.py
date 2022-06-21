#!/usr/bin/env python
""" This module propose several functions used to extract features from sequence of phases"""
from __future__ import division, absolute_import, generators, print_function, unicode_literals,\
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2017-2019 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import numpy as np
from jobmodel import mongodb_extractor


IO_DURATIONS_HISTO_BIN = 16


def extract_phases(seq, targeted_phase_type='C', tol=0):
    """Extract all the phases corresponding to targeted_phase_type in the sequence
    applying tolerance against outliers using the negative method.

    Args:
        seq (str): sequence of symbols representing the running phases
        targeted_phase_type (str): the symbols of the extracted phases
        tol (int): max size of the outliers phases that will be merged into targeted phases

    Returns:
        ndarray of bool where True represent a timeframe caracterise as targed_phases_type
    """
    seq_array = np.array(list(seq))
    seq_length = len(seq)

    # if there are not phase in the sequence, returns
    if seq_length == 0:
        return np.array([])

    # reversing problem, getting indices of all which is not targeted_phase_type
    negative_indices = np.where(seq_array != targeted_phase_type)[0]

    # if sequence contains only checkpoint phase:
    if not negative_indices.size and targeted_phase_type in seq_array:
        return np.ones(seq_length).astype(np.bool)

    # if the are not targeted_phase in the sequence, returns
    if not negative_indices.size:
        return np.array([])

    first = True
    negative_idx_phases = []
    current_phase = []

    for idx in negative_indices:
        if first:
            previous = idx
            current_phase.append(idx)
            first = False
            continue
        if previous + 1 == idx:
            previous = idx
            current_phase.append(idx)
        else:
            negative_idx_phases.append(current_phase)
            current_phase = [idx]
            previous = idx

    # adding the last negative phase
    negative_idx_phases.append(current_phase)

    # filtering the negative phases inferior to tolerance factor
    tolerated_negative_idx_phases = np.array([index for segment in negative_idx_phases\
                                               for index in segment if len(segment) > tol])
    targeted_phases = np.ones(seq_length).astype(np.bool)
    if tolerated_negative_idx_phases.size:
        targeted_phases[tolerated_negative_idx_phases] = False

    return targeted_phases


def get_phase_boundaries(seq, targeted_phase_type='C', tol=0):
    """Get the starting and ending indices of each targeted phases found in the sequence.
    Based on the negative phase extraction method, this function allow to get the phase
    boundaries once the tolerance have been applied to outliers.

    Args:
        seq (str): sequence of symbols representing the running phases
        targeted_phase_type (str): the symbols of the phases extacted to be bounded
        tol (int): max size of the outliers phases that will be merged into targeted phases

    Returns:
        list of tuple containing the (start, end) indices for each targeted phase
    """
    targeted_phases = extract_phases(seq, targeted_phase_type, tol)
    starts = []
    ends = []
    phase = []
    for i, phase in enumerate(targeted_phases):
        if i == 0:
            previous = phase
            if phase:
                starts.append(i)
        if phase != previous:
            if phase:
                starts.append(i)
            else:
                ends.append(i-1)
            previous = phase
    if phase:
        ends.append(i)

    boundaries = list(zip(starts, ends))

    return boundaries


def inter_phases_time(seq, targeted_phase_type='C', tol=0):
    """Compute the non-overlaping number of timeframes between targeted_phases
    Compute the number of timeframes between the end of a each consecutive phase
    (of type targeted_phase) and the next start of the same kind of phases
    it also include the timeframe number before the first phases occurency and
    after the last.

    Args:
        seq (str): the sequence of phases as symbols
        targeted_phase_type (str) : the symbols of the phases used to compute time intervals
        tol (int): max size of the outliers phases that will be merged into targeted phases

    Returns:
        ndarray of the intervals (in timeframe numbers)
    """
    targeted_phases_boundaries = np.array(get_phase_boundaries(seq, targeted_phase_type, tol))
    nb_bounds = targeted_phases_boundaries.shape[0]

    if targeted_phases_boundaries.size == 0:
        return []

    starts = np.zeros((nb_bounds+1))
    ends = np.zeros((nb_bounds+1))

    starts[:-1] = targeted_phases_boundaries[:, 0]
    ends[1:] = targeted_phases_boundaries[:, 1]

    starts[-1] = len(seq)

    inter_phases = (starts - ends) - 1

    return inter_phases


def rle(seq, sep='|'):
    """Implementation of the Run-length encoding (RLE) Algorithm
    A very simple form of lossless data compression for sequences where
    the same data value occurs in many consecutive data elements.
    Those data are stored as a single data value and count

    Args:
        seq (str): the sequence of phases as symbols
        sep (str): the separator used between phases in the resulting string

    Returns:
        the compact rle sequence, symboles + number of occurences, for each phases
    """
    sym = []
    occ = []
    count = 1
    prec = seq[0]
    for symbol in seq[1:]:
        if symbol == prec:
            count += 1
        else:
            sym.append(prec)
            occ.append(count)
            count = 1
            prec = symbol
    sym.append(prec)
    occ.append(count)
    assert len(sym) == len(occ), 'each symbol do not match with its occurency number'
    res = ''
    for symbol, occurence in zip(sym, occ):
        res += str(occurence) + symbol + sep
    return res[:-1]


def total_phase_volume(seq, data, targeted_phase_type='C', tol=0):
    """Compute the total volume read/write for each targeted phase
    Compute the total volume read/write for each targeted phase

    Args:
        seq (str): the sequence of phases as symbols
        data (dataframe): metrics data associated with the sequence
        targeted_phase (str) : the symbols of the phases used to compute
        tol (int): max size of the outliers phases that will be merged into targeted phases

    Returns:
        a list of tuples (index of begining phase, index of end phase, volume read in bytes,
        volume write in bytes)
    """
    total_volume_list = []
    phase_list = get_phase_boundaries(seq, targeted_phase_type, tol)

    for phase in phase_list:
        total_read = int(data.iloc[phase[0]:phase[1]+1]["bytesRead"].sum())
        total_write = int(data.iloc[phase[0]:phase[1]+1]["bytesWritten"].sum())
        total_volume_list.append((phase[0], phase[1], total_read, total_write))

    return total_volume_list


def build_feature_array(feature_df):
    """Transform and convert columns in data frame into 1D numpy array

    Args:
        feature dataframe

    Returns:
        1D numpy array
    """
    feature_arr = np.array([])
    for col in feature_df.columns:
        arr = feature_df[col].values[0] # histogram value
        feature_arr = np.concatenate((feature_arr, arr), axis=0)
    return feature_arr


def get_histogram_feature(database, jobid, norm="meanmax"):
    """Compute for each job a histogram feature in 1D numpy array

    Args:
        database : database connector
        jobid (int): job id in database
        norm: nomalization method (meanmax or zscore)

    Returns:
        a numpy array
    """
    connector = mongodb_extractor.MetricsDataBuilder(database, jobid)
    # get a dataframe of all histograms
    df_histo = connector.get_job_histograms(norm)
    # Generate feature vector from dataframe
    arr = build_feature_array(df_histo)
    #print(arr)
    return arr


def get_histogram_feature_matrix(database, jobids, norm=""):
    """Compute a histogram feature matrix for a list of job

    Args:
        database : database connector
        jobids (list): list of job id in database
        norm: nomalization method

    Returns:
        a 2D numpy array
    """
    # compute the first histogram feature
    arr = get_histogram_feature(database, jobids[0], norm)
    col = arr.shape[0]
    mat = np.empty(shape=[0, col])
    for jid in jobids:
        arr = get_histogram_feature(database, jid, norm)
        mat = np.vstack((mat, arr))
    return mat


def generate_bin_times():
    """Generate bin values for duration histogram

    Returns:
        bin_times (dict): the name range as key and correspondant interval list as value
    """

    n_bins = IO_DURATIONS_HISTO_BIN
    bin_times = {}
    for i in range(n_bins):
        if not i:
            bin_times['range'+str(i)] = [0, 4**(i)]

        else:
            bin_times['range'+str(i)] = [4**(i-1), 4**i]
    return bin_times


def time_and_hit_access(bin_times, bin_hard_hit, hist):
    """Extract the number of read IO in the hit bins and time spend to do read IO (except hit bins)
    from the histogram of read or write accesses.

    Args:
        bin_times (dict): dictionary of the ranges of access in the histogram
        bin_hard_hit (list): list of the ranges corresponding to hit accesses
        hist (pandas.series): pandas series of the raw data

    Returns:
        access_time (float): time spend to do read IO (except hit bins)
        hard_hit (float): number of read IO in the hit bins
    """
    if hist:
        hard_hit = sum([value for key, value in hist.items()
                        if key in bin_hard_hit])
        access_time = sum([bin_times[key][0]*value for key, value in hist.items()
                           if key not in bin_hard_hit])
    else:
        access_time = 0
        hard_hit = 0
    return access_time, hard_hit


def compute_save_time(row, bin_times, bin_hard_hit):
    """Compute an estimation of the time that could be saved using SRO from the histograms of read
    and write accesses.

    Args:
        row (pandas.series): pandas series of the raw data
        bin_times (dict): dictionary of the ranges of access in the histogram
        bin_hard_hit (list): list of the ranges corresponding to hit accesses

    Returns:
        save_time (float): time that could be saved using SRO
    """
    read_hist = row['read']
    read_access_time, hard_hit_read = time_and_hit_access(bin_times,
                                                          bin_hard_hit,
                                                          read_hist)

    write_hist = row['write']
    write_access_time, hard_hit_write = time_and_hit_access(bin_times,
                                                            bin_hard_hit,
                                                            write_hist)

    #number of total IO in the hit bins
    hard_hit = hard_hit_read + hard_hit_write
    #number of total IO in the hit bins
    access_time = read_access_time + write_access_time
    #compute the percent of accelerable IO (except hit IO)
    n_access = sum(row[['accelerated', 'accelPossible', 'accelImpossible']]) - hard_hit
    if n_access:
        ratio_accel = row['accelPossible']/n_access
    else:
        ratio_accel = 0

    #savable time in the timeframe
    save_time = ratio_accel*(access_time)/row['ioActiveProcessCount']
    return save_time


def compute_job_score(database, jobid):
    """Compute for each job a value based on a scoring function

    Args:
        database : database connector
        jobid (int): job id in database

    Returns:
        score (float): score value
        accel (float): percentage of accellerability
    """
    connector = mongodb_extractor.MetricsDataBuilder(database, jobid)

    # get a dataframe of selected metrics
    metrics = {'FileIOSummaryGw': ['accelerated', 'accelPossible', 'accelImpossible'],
               'IODurationsGw': ['read', 'write'],
               'ProcessSummaryGw': ['ioActiveProcessCount']}
    data = connector.select_data_from_db(metrics)

    # compute job time from job metadata
    job_item = database['JobItem'].loc[jobid][['startTime', 'endTime']]
    job_time = (job_item['endTime'] - job_item['startTime']).seconds

    bin_times = generate_bin_times()
    total_save_time = 0
    #bin where IO are hit for sure
    bin_hard_hit = ['range'+str(i) for i in range(3)]

    for _, row in data.iterrows():
        if row['ioActiveProcessCount']:
            save_time = compute_save_time(row, bin_times, bin_hard_hit)
        else:
            save_time = 0

        #savable time in the job
        total_save_time += save_time

    accel = total_save_time/(job_time*1000000)
    score = -(1/(3*accel+(np.sqrt(10)/10))**2)+10

    return score, accel*100


def compute_distance_job_histograms(database, jobid1, jobid2, norm="meanmax"):
    """Compute distance euclidian for 2 jobs from size and duration histograms

    Args:
        database : database connector
        jobid (int): job id in database
        norm : normalization method (meanmax or zscore)

    Returns:
        dist (float): distance value of 2 jobs
    """
    # get size and duration histogram for each job
    connector = mongodb_extractor.MetricsDataBuilder(database, jobid1)
    hist1 = connector.get_job_histograms(norm)

    connector = mongodb_extractor.MetricsDataBuilder(database, jobid2)
    hist2 = connector.get_job_histograms(norm)

    dist = 0
    # compute euclidian distance for each histogram
    for col in hist1.columns:
        arr1 = hist1[col].values[0]
        arr2 = hist2[col].values[0]
        dist += np.linalg.norm(np.array(arr1)-np.array(arr2))

    return dist
