#!/usr/bin/env python
""" This module proposes functions to perform visualization of jobmodel results.
Could it be useful to create a dedicated common visualization library for all IOanalytics code ?
It could permits to remove matplotlib dependency on our productized libraries and improve the
coverage ratio because these visualization features are not tested.
"""

from __future__ import division, absolute_import, generators, print_function, unicode_literals,\
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2019 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import pylab
import matplotlib.pyplot as plt
import numpy as np

pylab.rcParams['figure.figsize'] = (60, 25)

COLOR_TABLE = {'I': 50,
               'A': 20,
               'C': 190,
               'S': 140,
               'E': 230,
               'U': 255,
               '-': 255,
               'match': 0,
               'mismatch': 255}


def show_sequences_as_image(sequences, labels):
    """Format the alignment prettily into an image

    Args:
        sequences: list of sequences
        labels: list of sequence labels
    """
    assert sequences != [], 'list of sequences should not be empty'
    max_length = max([len(s) for s in sequences])
    nb_seq = len(sequences)
    img = np.zeros((max_length, nb_seq))
    for i, seq in enumerate(sequences):
        for j, sym in enumerate(seq):
            img[j, i] = COLOR_TABLE[sym]

    plt.imshow(img.T, aspect='auto', interpolation='none', origin='lower', cmap='nipy_spectral',
               vmin=0, vmax=255)

    plt.gca().set_yticks(range(len(labels)))
    plt.gca().set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    plt.gca().set_yticklabels(labels)
    plt.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.show()


def format_alignment_as_image(align1, align2):
    """Format the alignment prettily into an image."""
    img = np.zeros((len(align1), 3))
    for i, elem in enumerate(align1):
        img[i, 0] = COLOR_TABLE[elem]
        img[i, 1] = COLOR_TABLE[elem]
        if elem == align2[i]:
            img[i, 2] = COLOR_TABLE['match']
        else:
            img[i, 2] = COLOR_TABLE['mismatch']

    plt.imshow(img.T, aspect='auto', interpolation='none', origin='lower', cmap='nipy_spectral')
    plt.show()


def format_distance_matrix(distance_matrix, labels, step=2, str_format=':4.2f'):
    """Format the distance matrix to be displayed.

    Args:
        distance_matrix (numpy array): ndarray to be displayed.
        labels (list): labels of the sequences.
        step (int): number of space separator between columns.
        str_format (string): formatting used to print numerical values.
    """
    assert distance_matrix.ndim == 2, 'distance_matrix has wrong dimensionality : \
distance_matrix.ndim = %i' % distance_matrix.ndim

    str_format = '{' + str_format + '}'
    mat = [[] for i in range(distance_matrix.shape[0])]
    header = []

    # formats the first column od labels
    max_lab_length = max([len(lab) for lab in labels])
    for j in range(distance_matrix.shape[1]):
        str_lab = str(labels[j])
        mat[j].append(' ' * (max_lab_length - len(str_lab)) + str_lab + ' ' * step + '|')
    corner = ' ' * max_lab_length + ' ' * step + '|'
    header.append(corner)

    # formats the core of the matrix
    for j in range(distance_matrix.shape[1]):
        str_lab = str(labels[j])
        max_len = max(max([len(str_format.format(val)) for val in distance_matrix[:, j]]),
                      len(str_lab))
        header.append(' ' * step + ' ' * (max_len - len(str_lab)) + str_lab)
        for i in range(distance_matrix.shape[0]):
            str_val = str_format.format(distance_matrix[i, j])
            mat[j].append(' ' * step + ' ' * (max_len - len(str_val)) + str_val)

    # prints the built matrix
    header = ''.join(header)
    print(header)
    separator = list('-' * len(header))
    separator[len(corner) - 1] = '+'
    print(''.join(separator))
    for i in range(distance_matrix.shape[0]):
        print(''.join(mat[i]))
