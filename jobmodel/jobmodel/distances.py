#!/usr/bin/env python
""" This module proposes classes and functions to perform distance computation between time series.
"""

from __future__ import division, absolute_import, generators, print_function, unicode_literals,\
                       with_statement, nested_scopes  # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2019 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import numpy as np
from scipy.stats import distributions

from ioanalyticstools.time_series_manipulation import stack_timeseries
from ioanalyticstools.statistical_tests.goodness_of_fit import cvm_pval
from ioanalyticsextensions.cdtw import c_dtw_float, c_dtw_int
from jobmodel.phase_comparator import pairwise_alignment

# The following part of code define several distance functions and the related wrapper(s) to define
# a common interface format for the arguments and the output of the distance functions.
# These functions are designed to permit usage modularity with the 'DistanceJob' class of the
# 'job_model' module.
#
# The prototype of these functions is:
#
# def dist_custom(ts1, ts2, *args, **kwargs):
#     """
#     Args:
#         ts1 (numpy array): the first time-series array
#         ts2 (numpy array): the second time-series array
#         *args: additional positional arguments
#         **kwargs: additional keyword arguments
#
#     Return:
#         a scalar value of distance between ts1 and ts2
#     """
#     return distance_fct(*args, **kwargs)[0]


# The 'seq_alignment_distance' function is presented here, with its related wrapper
# 'seq_alignment_dist'.

def max_score(sequence, score_matrix=None):
    """Computes the maximum similarity score for a given sequence (the sequence compared to itself)
    using an optional score_matrix if passed as argument

    Args:
        sequence (str): sequence of symbol.
        score_matrix (dict): dictionary containing the scores associated to the matching
        of two symbols.

    Returns:
        score (int): the maximum similarity score a sequence can obtain (identity case).
    """
    score = 0
    for sym in sequence:
        if score_matrix:
            score += score_matrix[(sym, sym)]
        else:
            score += 1
    return score


def seq_alignment_distance(sequence1, sequence2, score_matrix=None):
    """Computes the pairwise distance of 2 sequences using the 'Bio.pairwise2' feature based on
    sequence alignment. The score_matrix, if passed as argument, weight alignment of symbol pairs.

    Args:
        sequence1 (str): the first sequence of symbol.
        sequence2 (str): the second sequence of symbol.
        score_matrix (dict): dict of the weight by symbol pair.

    Returns:
        (float) the pairwise sequence alignment distance.
    """
    i_score_max = max_score(sequence1, score_matrix)
    j_score_max = max_score(sequence2, score_matrix)
    score_max = max(i_score_max, j_score_max)
    align = pairwise_alignment(sequence1,
                               sequence2,
                               score_matrix=score_matrix,
                               nb_disp=0)
    _, _, score, _, _ = align[0]

    if not score_max:
        return 0

    return 1 - score/score_max


def seq_alignment_dist(ts1, ts2, score_matrix=None):
    """Wraps the 'pairwise_distance' function to compute the Dynamic Time Wrapping (DTW) distance
    between to time-series.

    Args:
        ts1 (numpy array): the first time-series array
        ts2 (numpy array): the second time-series array
        score_matrix (dict): dict of the weight by symbol pair.

    Return:
        the pairwise sequence alignment distance between ts1 and ts2
    """
    return seq_alignment_distance(ts1, ts2, score_matrix=score_matrix)


# The 'ks_distance' function is presented here, with its related wrappers 'ks_delta_dist' and
# 'ks_pvalue_dist'.


def ks_distance(pdf1, pdf2):
    """Computes the distance between 2 distribution from the pValue of the Kolmogorov-Smirnov
    goodness of fit hypothesis test. The both input PDFs will be considered such that them have
    same bins size and the first bin is the same (i.e. : bin0 is first timeframe).

    Args:
        pdf1 (list or numpy array): the first empirical probability distribution to be compared.
        pdf2 (list or numpy array): the second empirical probability distribution to be compared.

    Returns:
        delta (float): the ks statistics, the probability distribution distance between the CDFs.
        pValue (float): the ks p-value.
    """
    pdf1 = np.array(pdf1, dtype=np.uint64)
    pdf2 = np.array(pdf2, dtype=np.uint64)
    # test if there is none event in a pdf
    if not pdf1.any() and not pdf2.any():
        return 0, 1
    if not pdf1.any() or not pdf2.any():
        return 1, 0

    n_samp1 = pdf1.sum()
    n_samp2 = pdf2.sum()

    pdf_padded = stack_timeseries([pdf1, pdf2])

    cdf1 = np.cumsum(pdf_padded[0])/(1.0*n_samp1)
    cdf2 = np.cumsum(pdf_padded[1])/(1.0*n_samp2)

    delta = np.max(np.absolute(cdf1-cdf2))
    # Note: delta absolute not signed distance
    coef = np.sqrt((n_samp1*n_samp2/(n_samp1+n_samp2)).astype(np.float64))
    try:
        p_value = distributions.kstwobign.sf((coef + 0.12 + 0.11 / coef) * delta)
    except ValueError:
        p_value = 1.0
    return delta, p_value


def ks_delta_dist(ts1, ts2):
    """Wraps the 'ks_distance' function to compute the ks statistics as a distance between to
    time-series.

    Args:
        ts1 (numpy array): the first time-series array
        ts2 (numpy array): the second time-series array

    Return:
        ks statistics as a distance between ts1 and ts2
    """
    return ks_distance(ts1, ts2)[0]


def ks_pvalue_dist(ts1, ts2):
    """Wraps the 'ks_distance' function to compute the ks pvalue as a distance between to
    time-series.

    Args:
        ts1 (numpy array): the first time-series array
        ts2 (numpy array): the second time-series array

    Return:
        ks pvalue as a distance between ts1 and ts2
    """
    return ks_distance(ts1, ts2)[1]


# The 'cvm_distance' function is presented here, with its related wrappers 'cvm_stat_dist' and
# 'cvm_pvalue_dist'.

def cvm_distance(pdf1, pdf2):
    """Computes the distance between 2 distribution PDFs from the criterion of the Cramer-von Mises
    goodness of fit hypothesis test.

    Args:
        pdf1 (list or numpy array): the first empirical probability distribution to be compared.
        pdf2 (list or numpy array): the second empirical probability distribution to be compared.

    Returns:
        statistics (float): the cvm statistics, the distance related to area difference between
        the two CDFs.
        pValue (float): the cvm p-value.
    """
    pdf1, pdf2 = map(np.asarray, (pdf1, pdf2))
    # test if there is none event in a pdf
    if not pdf1.any() and not pdf2.any():
        return 0, 1

    if not pdf1.any() or not pdf2.any():
        if not pdf1.any():
            pdf = pdf2
        else:
            pdf = pdf1

        n_samp = pdf.sum()
        cdf = np.cumsum(pdf)/(1.0*n_samp)
        statistics = np.sum(cdf**2 * pdf)

        return statistics, 0

    else:
        n_samp1 = pdf1.sum()
        n_samp2 = pdf2.sum()

        pdf_padded = stack_timeseries([pdf1, pdf2])

        cdf1 = np.cumsum(pdf_padded[0])/(1.0*n_samp1)
        cdf2 = np.cumsum(pdf_padded[1])/(1.0*n_samp2)

        omega = np.sum((cdf1-cdf2)**2 * np.sum(pdf_padded, axis=0))
        statistics = n_samp1 * n_samp2 * omega / (n_samp1 + n_samp2)**2

        # if samples are just single values the cvm test could not be performed
        if n_samp1 == 1 and n_samp2 == 1:
            if np.array_equal(pdf1, pdf2):
                return statistics, 1
            return statistics, 0

        p_value = cvm_pval(n_samp1, n_samp2, statistics)

        return statistics, p_value


def cvm_stat_dist(ts1, ts2):
    """Wraps the 'cvm_distance' function to compute the cvm statistics as a distance between to
    time-series.

    Args:
        ts1 (numpy array): the first time-series array
        ts2 (numpy array): the second time-series array

    Return:
        cvm statistics as a distance between ts1 and ts2
    """
    return cvm_distance(ts1, ts2)[0]


def cvm_pvalue_dist(ts1, ts2):
    """Wraps the 'cvm_distance' function to compute the cvm pvalue as a distance between to
    time-series.

    Args:
        ts1 (numpy array): the first time-series array
        ts2 (numpy array): the second time-series array

    Return:
        cvm pvalue as a distance between ts1 and ts2
    """
    return cvm_distance(ts1, ts2)[1]


# The wrappers 'dtw_dist' and 'dtw_dist_int' is presented here. The distance functions used are
# respectively 'c_dtw_float' and 'c_dtw_int' from the custom cython 'cdtw' extension from the
# ioanalyticstools module (actually the ioanalyticsextensions module)

def dtw_dist(ts1, ts2):
    """Wraps the 'c_dtw_float' function to compute the Dynamic Time Wrapping (DTW) distance between
    to time-series of float values.

    Args:
        ts1 (numpy array): the first time-series array.
        ts2 (numpy array): the second time-series array.

    Return:
        the DTW distance between ts1 and ts2.
    """
    return c_dtw_float(ts1.astype('float64'), ts2.astype('float64'))


def dtw_dist_int(ts1, ts2):
    """Wraps the 'c_dtw_int' function to compute the Dynamic Time Wrapping (DTW) distance between
    to time-series of int values.

    Args:
        ts1 (numpy array): the first time-series array.
        ts2 (numpy array): the second time-series array.

    Return:
        the DTW distance between ts1 and ts2.
    """
    return c_dtw_int(ts1, ts2)
