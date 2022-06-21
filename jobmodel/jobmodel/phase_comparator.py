#!/usr/bin/env python
""" This module proposes functions to compare pairwise sequences of operating phases
"""
from __future__ import division, absolute_import, generators, print_function, unicode_literals,\
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2017 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import numpy as np

from Bio import pairwise2
from Bio.pairwise2 import format_alignment


SCORE_MATRIX_FT = {('I', 'I'): 2,
                   ('I', 'A'): -1,
                   ('I', 'C'): -1,
                   ('I', 'S'): -1,
                   ('I', 'E'): -1,
                   ('A', 'A'): 2,
                   ('A', 'C'): -1,
                   ('A', 'S'): 1,
                   ('A', 'E'): -1,
                   ('C', 'C'): 2,
                   ('C', 'S'): -2,
                   ('C', 'E'): 1,
                   ('S', 'S'): 2,
                   ('S', 'E'): -3,
                   ('E', 'E'): 2}


SCORE_MATRIX_LV = {('A', 'A'): 5,
                   ('A', 'C'): 4,
                   ('A', 'E'): 3,
                   ('A', 'S'): 3,
                   ('C', 'C'): 5,
                   ('C', 'E'): 2,
                   ('C', 'S'): 2,
                   ('E', 'E'): 5,
                   ('I', 'A'): -5,
                   ('I', 'C'): -5,
                   ('I', 'E'): -5,
                   ('I', 'I'): 5,
                   ('I', 'S'): -5,
                   ('S', 'E'): -3,
                   ('S', 'S'): 5}


def pairwise_alignment(seq1, seq2, score_matrix=None, nb_disp=None):
    """Pairwise sequence alignment using a dynamic programming algorithm."""
    if score_matrix is None:
        alignments = pairwise2.align.globalxx(seq1, seq2)
    else:
        alignments = pairwise2.align.globaldx(seq1, seq2, score_matrix)
    nb_align = len(alignments)
    if nb_disp is None or nb_disp > nb_align:
        nb_disp = nb_align
    for i in range(nb_disp):
        print(format_alignment(*alignments[i]))
    return alignments


def sseq_to_str(sseq):
    """ numpy s* to str casting"""
    str_repr = ''.join([p for p in list(sseq)])
    return str_repr


def convert(pseq):
    """ convert initial format using symbol alphabet."""
    print("phases found :")
    print(np.unique(pseq))
    print("converting sequence to symbol using following table")
    cvrt = {'IO':'A', 'IOInactive':'I', 'checkpoint':'C', 'end':'E', 'start':'S'}
    print(cvrt)
    sseq = np.zeros_like(pseq)
    for phase in np.unique(pseq):
        try:
            symbol = cvrt[phase]
        except:
            raise Exception('Cannot convert a phase which is not defined in the cvrt table')
        sseq[np.where(pseq == phase)] = symbol
    sseq = sseq_to_str(sseq)
    return sseq
