""" IOPA jobmodel unittests module """

from __future__ import division, absolute_import, generators, print_function, unicode_literals, \
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2019 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import numpy as np
import unittest
from ioanalyticsextensions.cdtw import c_dtw_float, c_dtw_int
from jobmodel.distances import max_score, seq_alignment_distance, ks_distance, cvm_distance, \
                               seq_alignment_dist, ks_delta_dist, ks_pvalue_dist, cvm_stat_dist, \
                               cvm_pvalue_dist, dtw_dist, dtw_dist_int


class TestMaxScore(unittest.TestCase):
    """ TestCase used to test the 'max_score' function of the 'distances' module."""
    def setUp(self):
        self.symbol_seq = 'AAAIIIAAA'

    def test_max_score_wo_matrix(self):
        """ Test the 'max_score' returned value is 0, if all weights in the score matrix
        are 0."""
        score = max_score(self.symbol_seq)
        self.assertEqual(len(self.symbol_seq), score)

    def test_max_score_0(self):
        """ Test the 'max_score' returned value is 0, if all weights in the score matrix
        are 0."""
        score_mat = {('I', 'I'): 0,
                     ('I', 'A'): 0,
                     ('A', 'A'): 0}
        score = max_score(self.symbol_seq, score_mat)
        self.assertEqual(0, score)

    def test_max_score(self):
        """ Test the 'max_score' returned value is the length of the sequence, if all weight
        in the score matrix are 1."""
        score_mat = {('I', 'I'): 1,
                     ('I', 'A'): 1,
                     ('A', 'A'): 1}
        score = max_score(self.symbol_seq, score_mat)
        self.assertEqual(len(self.symbol_seq), score)

    def test_max_score_weighted(self):
        """ Test the 'max_score' returned value, if the score matrix have weights."""
        score_mat = {('I', 'I'): 5,
                     ('I', 'A'): 1,
                     ('A', 'A'): 3}
        score = max_score(self.symbol_seq, score_mat)
        self.assertEqual(33, score)


class TestSeqAlignmentDistance(unittest.TestCase):
    """ TestCase used to test the Sequence Alignment related functions of the 'distances' module."""
    def setUp(self):
        self.symbol_seq1 = 'AAAAAA'
        self.symbol_seq2 = 'AAAIIIAAA'
        self.symbol_seq3 = 'IIIII'
        self.symbol_seq4 = 'AAIAIIAAIIAI'

    def test_seq_alignment_distance_0(self):
        """ Test the 'seq_alignment_distance' returned value is 0, if matching symbol weights in the
        score matrix are 0."""
        score_mat = {('I', 'I'): 0,
                     ('I', 'A'): 10,
                     ('A', 'A'): 0}
        score = seq_alignment_distance(self.symbol_seq1, self.symbol_seq2, score_matrix=score_mat)
        self.assertEqual(0, score)

    def test_seq_alignment_distance_1(self):
        """ Test the 'seq_alignment_distance' returned value is 1 for 2 different sequences, if
        un-matching symbol weights in the score matrix are 0."""
        score_mat = {('I', 'I'): 5,
                     ('I', 'A'): 0,
                     ('A', 'A'): 3}
        score = seq_alignment_distance(self.symbol_seq1, self.symbol_seq3, score_matrix=score_mat)
        self.assertEqual(1, score)

    def test_seq_alignment_distance_weighted(self):
        """ Test the 'seq_alignment_distance' returned value is 1 for 2 different sequences, if all
        weight of corresponding symbols in the score matrix are 1."""
        score_mat = {('I', 'I'): 10,
                     ('I', 'A'): 2,
                     ('A', 'A'): 15}
        score = seq_alignment_distance(self.symbol_seq1, self.symbol_seq4, score_matrix=score_mat)
        self.assertEqual(0.4, score)

    def test_seq_alignment_dist_wrapper(self):
        """ Test the wrapper 'seq_alignment_dist' returned same value as the original function
        'seq_alignment_distance'."""
        score_mat = {('I', 'I'): 10,
                     ('I', 'A'): 2,
                     ('A', 'A'): 15}
        score = seq_alignment_distance(self.symbol_seq1,
                                       self.symbol_seq4,
                                       score_matrix=score_mat)
        score_wrapper = seq_alignment_dist(self.symbol_seq1,
                                           self.symbol_seq4,
                                           score_matrix=score_mat)
        self.assertEqual(score, score_wrapper)


class TestKSDistance(unittest.TestCase):
    """ TestCase used to test the KS related and functions of the 'distances' module."""
    def setUp(self):
        self.pdf1 = np.array([2, 2, 2])
        self.pdf2 = np.array([0, 0, 0, 0, 3])
        self.pdf3 = np.array([0, 1, 3, 0, 2])

    def test_ks_distance_0(self):
        """ Test the 'ks_distance' returned matching values for identical samples (expected 'delta'
        and 'pval' values come from the scipy implementation of the test 'ks_2samp')."""
        delta, pval = ks_distance(self.pdf1, self.pdf1)
        self.assertEqual(0, delta)
        self.assertEqual(1, pval)

    def test_ks_distance_1(self):
        """ Test the 'ks_distance' returned un-matching values for different samples (expected
        'delta' and 'pval' values come from the scipy implementation of the test 'ks_2samp')."""
        delta, pval = ks_distance(self.pdf1, self.pdf2)
        self.assertEqual(1, delta)
        self.assertAlmostEqual(0, pval, 1)

    def test_ks_distance_other(self):
        """ Test the 'ks_distance' returned expected values for samples (expected 'delta' and 'pval'
        values come from the scipy implementation of the test 'ks_2samp')."""
        delta, pval = ks_distance(self.pdf1, self.pdf3)
        self.assertEqual(0.5, delta)
        self.assertAlmostEqual(0.3, pval, 1)

    def test_ks_delta_dist_wrapper(self):
        """ Test the wrapper 'ks_delta_dist' returned same delta value as the original function
        'ks_distance'."""
        delta, _ = ks_distance(self.pdf1, self.pdf3)
        dist = ks_delta_dist(self.pdf1, self.pdf3)
        self.assertEqual(delta, dist)

    def test_ks_pvalue_dist_wrapper(self):
        """ Test the wrapper 'ks_pvalue_dist' returned value same pvalue value as the original
        function 'ks_distance'."""
        _, pvalue = ks_distance(self.pdf1, self.pdf3)
        dist = ks_pvalue_dist(self.pdf1, self.pdf3)
        self.assertEqual(pvalue, dist)


class TestCVMDistance(unittest.TestCase):
    """ TestCase used to test the CvM related functions of the 'distances' module."""
    def setUp(self):
        self.pdf1 = np.array([1, 2, 0, 2, 1])
        self.pdf2 = np.array([0, 1, 1, 1, 3])

    def test_cvm_distance_0(self):
        """ Test the 'cvm_distance' returned matching values for identical samples (expected
        'statistics' and 'pval' values should be verified with external implementation)."""
        statistics, pval = cvm_distance(self.pdf1, self.pdf1)
        self.assertEqual(0, statistics)
        self.assertEqual(1, pval)

    def test_cvm_distance_other(self):
        """ Test the 'cvm_distance' returned expected values for samples (expected 'statistics' and
        'pval' values should be verified with external implementation)."""
        statistics, pval = cvm_distance(self.pdf1, self.pdf2)
        self.assertAlmostEqual(0.18, statistics, 2)
        self.assertAlmostEqual(0.3, pval, 1)

    def test_cvm_distance_empty(self):
        """ Test the 'cvm_distance' returned matching values for empty samples."""
        pdf1 = np.array([0, 0])
        statistics, pval = cvm_distance(pdf1, pdf1)
        self.assertEqual(0, statistics)
        self.assertEqual(1, pval)

    def test_cvm_distance_pdf1_empty(self):
        """ Test the 'cvm_distance' returned unmatching values for one empty samples."""
        pdf1 = np.array([0, 0])
        statistics, pval = cvm_distance(pdf1, self.pdf2)
        self.assertAlmostEqual(3.39, statistics, 2)
        self.assertEqual(0, pval)

    def test_cvm_distance_pdf2_empty(self):
        """ Test the 'cvm_distance' returned unmatching values for one empty samples."""
        pdf2 = np.array([0, 0])
        statistics, pval = cvm_distance(self.pdf1, pdf2)
        self.assertAlmostEqual(2.92, statistics, 2)
        self.assertEqual(0, pval)

    def test_cvm_distance_one_elem(self):
        """ Test the 'cvm_distance' returned unmatching values for one element samples."""
        pdf1 = np.array([0, 1, 0, 0])
        pdf2 = np.array([1, 0, 0])
        statistics, pval = cvm_distance(pdf1, pdf2)
        self.assertAlmostEqual(0.25, statistics, 2)
        self.assertEqual(0, pval)

    def test_cvm_distance_same_one_elem(self):
        """ Test the 'cvm_distance' returned matching values for identical one element samples."""
        pdf1 = np.array([0, 1, 0, 0])
        statistics, pval = cvm_distance(pdf1, pdf1)
        self.assertEqual(0, statistics, 2)
        self.assertEqual(1, pval)

    def test_cvm_stat_dist_wrapper(self):
        """ Test the wrapper 'cvm_delta_dist' returned same statistics value as the original
        function 'cvm_distance'."""
        statistics, _ = cvm_distance(self.pdf1, self.pdf2)
        dist = cvm_stat_dist(self.pdf1, self.pdf2)
        self.assertEqual(statistics, dist)

    def test_cvm_pvalue_dist_wrapper(self):
        """ Test the wrapper 'cvm_pvalue_dist' returned value same pvalue value as the original
        function 'cvm_distance'."""
        _, pvalue = cvm_distance(self.pdf1, self.pdf2)
        dist = cvm_pvalue_dist(self.pdf1, self.pdf2)
        self.assertEqual(pvalue, dist)


class TestDTWDistance(unittest.TestCase):
    """ TestCase used to test the 'dtw_dist' functions of the 'distances' module."""

    def test_dtw_dist_wrapper(self):
        """ Test the wrapper 'dtw_dist' returned value in the same distance value as the original
        function 'c_dtw_float'."""
        ts_1_f = np.array([1.3, 2., 0.1, 2.5, 1.58, 8., 5.7])
        ts_2_f = np.array([0.7, 1., 1.88, 1.33, 3.])
        origin_dist = c_dtw_float(ts_1_f, ts_2_f)
        dist = dtw_dist(ts_1_f, ts_2_f)
        self.assertEqual(origin_dist, dist)

    def test_dtw_dist_int_wrapper(self):
        """ Test the wrapper 'dtw_dist_int' returned value in the same distance value as the original
        function 'c_dtw_int'."""
        ts_1 = np.array([1, 2, 0, 2, 1, 8, 5], dtype=np.int64)
        ts_2 = np.array([0, 1, 1, 1, 3], dtype=np.int64)
        print(ts_1.dtype)
        origin_dist = c_dtw_int(ts_1, ts_2)
        dist = dtw_dist_int(ts_1, ts_2)
        self.assertEqual(origin_dist, dist)


if __name__ == '__main__':
    unittest.main(verbosity=2)
