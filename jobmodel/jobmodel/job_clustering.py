#!/usr/bin/env python
""" This module proposes functions to perform clustering on a set of jobs from time series metrics
or from an analysis of their sequence of operating phases (symbols).
"""

from __future__ import division, absolute_import, generators, print_function, unicode_literals,\
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2017-2019 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import warnings
import numpy as np

from scipy.stats import ks_2samp

from ioanalyticstools.time_series_manipulation import binning
from ioanalyticstools.external.neighbor_joining import distance_tree
from ioanalyticstools.statistical_tests.goodness_of_fit import cvm_2samples
from jobmodel.distances import seq_alignment_distance, cvm_distance, ks_distance


def compute_distance_matrix(sequences, type_dist='pairwise', symmetric=True, verbosity=1,
                            score_matrix=None, granularity=1):
    """ Build the matrix of distance (pairwise distance between each pair of sequence) of
    a series of sequences from the score_matrix if passed as argument
    The both input PDFs will be considered such that they have same bins size and the first bin
    is the same (i.e. : bin0 is first timeframe)

    Args:
        sequences: list of sequences to be compared
        type_dist: type of distance to be used [pairwise | [KS-CvM]-[delta-pvalue]-[time-seq]]
        symmetric: boolean flag to indicate the score of A vs B and B vs A are equals
        verbosity: display distance result on stdout if activated (default)
        score_matrix: matrix of the weight by symbol matching (optional)
        granularity: binning factor reducing sequences along the time axis (optional)

    Returns:
        mat: ndarray to store all the pairwise distance
    """
    if granularity != 1:
        if type_dist == 'pairwise':
            warnings.warn("Warning: Granularity management is not available for the pairwise"
                          " distance computation. \"granularity\" forced to 1.", UserWarning)
        else:
            sequences = binning_mat(sequences, granularity)

    card = len(sequences)
    mat = np.zeros((card, card))
    pre_calc = {}
    for i, seq_i in enumerate(sequences):
        for j, seq_j in enumerate(sequences):
            if i == j:
                mat[i, j] = 0
                continue
            if symmetric and (j, i) in pre_calc:
                distance = pre_calc[(j, i)]
                mat[i, j] = distance
                continue

            if type_dist == 'pairwise':
                distance = seq_alignment_distance(seq_i, seq_j, score_matrix=score_matrix)

            elif type_dist == 'KS-pvalue-time':
                _, distance = ks_distance(seq_i, seq_j)
                distance = 1 - distance
            elif type_dist == 'KS-delta-time':
                distance, _ = ks_distance(seq_i, seq_j)
            elif type_dist == 'CvM-delta-time':
                distance, _ = cvm_distance(seq_i, seq_j)
            elif type_dist == 'CvM-pvalue-time':
                _, distance = cvm_distance(seq_i, seq_j)
                distance = 1 - distance

            elif type_dist == 'KS-pvalue-seq':
                _, distance = ks_2samp(seq_i, seq_j)
                distance = 1 - distance
            elif type_dist == 'KS-delta-seq':
                distance, _ = ks_2samp(seq_i, seq_j)
            elif type_dist == 'CvM-delta-seq':
                distance, _ = cvm_2samples(seq_i, seq_j)
            elif type_dist == 'CvM-pvalue-seq':
                _, distance = cvm_2samples(seq_i, seq_j)
                distance = 1 - distance

            else:
                raise NameError('Fatal : Unavailable type of distance computation',
                                '\'{}\'.'.format(type_dist))

            if verbosity:
                print('Distance between sequence', i, 'and', j, '=', distance)

            distance = np.abs(distance)
            mat[i, j] = distance
            pre_calc[(i, j)] = distance

    return mat


def binning_mat(mat, order):
    """ Function allowing to bin a list of ndarray by a given factor (order)
    This function sum all N contiguous elements of each array (N being the order) into bins and add
    a final bin as the sum of the remaining elements in the array.

    Args:
        mat (ndarray): numpy ndarray of numpy ndarray being binned
        order (int): the number of elements constituting the bins

    Returns:
        binned_mat (ndarray): a numpy ndarray the binned arrays
    """
    binned_mat = []
    for seq in mat:
        binned_mat.append(binning(seq, order))
    return binned_mat


def build_distance_tree(distance_matrix, labels, diff_scale):
    """ Build the clustering tree (binary tree) from the matrix of distance scores

    Args:
        distance_matrix: ndarray of pairwise distance
        labels: labels of the sequences
        diff_scale: a scale factor to display the tree

    Returns:
        tree: binary tree (from neighbor_joining) that represents the job classification
    """
    forest = np.array([distance_tree.Leaf(l) for l in labels])
    tree = distance_tree.neighborJoin(distance_matrix*diff_scale, forest)
    return tree
