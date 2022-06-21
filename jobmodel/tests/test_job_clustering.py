""" IOPA jobmodel unittests module """

from __future__ import division, absolute_import, generators, print_function, unicode_literals, \
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2017 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import unittest
import numpy as np

from jobmodel.job_clustering import compute_distance_matrix, build_distance_tree
from jobmodel.phase_comparator import SCORE_MATRIX_FT
from ioanalyticstools.external import neighbor_joining as nj


class TestJobClustering(unittest.TestCase):
    """ TestCase used to test 'job_clustering' module."""
    def setUp(self):
        sequence0 = 'SSSSSSAAAIIIAAAIIIACCCCCAAIIAAAIIAIAIAIIAAIIACCCCIIAAAIIIAIAIAIAIIAAIIAIAIACCC\
CCIAIAIAIAIAIAIEEEEEEE'
        sequence1 = 'SSSSSAAAAIIIAAAIAAACCCCCCAIIAAAIIAIAIAIIAIACCCCCIIAAIIIAIAIAIAIIAAIIAIAIAIAIAI\
AIAIAIAIAIAIAIEEEEEEEE'
        sequence2 = 'SSSSSSSAAAIAAIIIACCCCCAAIIAAAIICCCCCAIAIAIIAAIIACCCCIIAAAIIIAIAICCCCCCCAIIIAIA\
IACCCCCIAIAIAIAIEEEEEEE'
        sequence3 = 'SSSSSSAAAIIIAAAIIIAIIAIAAAIIAAAIIAIAIAIIAAIIACCCCCCIIAAAIIIAIAIAIAIIAAIIAIAIAC\
IAIAIAIAIAIAIAIAIEEEEEEE'
        sequence4 = 'SSSSSSAAAIIIAAAIIIACCCAAIIAAAIIIAIAIAIIAAIIACCCCIIAAAIIIAIAIIIAIAAIIAIAIACCCCC\
IAIAIAIAIAIAIEEEEEEEEE'
        sequence5 = 'SSSSSSSSAAAIIIAAAIIACCCCCCCAAIIAAAIIAAAIIACCCCCCCIIAAAIIIAIAIAIAIIAIAIACCCCCCC\
CIAIAIAIAIAIAIEEEEEEE'
        sequence6 = 'SSSSSSSSIIAAAIIIAIAAIAIAAAIIAAAIIAIAIAIIAAIIACCCCCCCCCCIIAAAIIIAIAIAIAIIAAIIAI\
AIAIAIAIAIAIAIAIEEEEEEEE'
        sequence7 = 'SSSSSSAAAIIIAACCCCCCCAAIIAAAIIAIACCCCCIAIIAAIIACCCCIIAAAIIIAIAICCCCCCAIIAIAIAC\
CCCCIAIAIAIAIAIAIEEEEEEE'
        sequence8 = 'SSSSSIIAAIIAAAIIIACCCCCCCCCAAIAIAIAIAIAIAIIAAIIACCCCCCCCIIAAAIIIAIAIAIAIIAAIIA\
IAIAIAIAIAIAIAIAIEEEEEEE'
        sequence9 = 'SSSSSSAAIIIAAAIIIAIAIAIAAAIIAAAIIAIAIAIIAAIIAIAIAIAIIAAIIIAIAIAIAIIAAIIAIAIAIA\
IIAAIAIAIAIAIAIEEEEEEEE'

        seqa = np.array([12743371, 25124371, 32280818, 29805552, 32270220, 32263268, 32260656,
                         32243900, 29764240, 32243648, 22438673, 10302915, 554377, 8967398,
                         17614651, 20100940, 10049498, 20095668, 20097112, 10043044, 10043000,
                         17492136, 12637796, 10037926, 20100184, 10050824, 12532888, 7569780,
                         17491544, 12628468, 10042068, 15008344, 15106712, 9930432, 10136764,
                         10034724, 14997736, 14996260, 10133716, 10035180, 14988184, 15083224,
                         20060972, 17472148, 17570924, 15090756, 20058932, 20050400, 14986408,
                         15086584, 17466272, 12941888, 19678288, 14990092, 20028952, 15079336,
                         20030136, 19924896, 20104148, 30020000, 30009956, 30006388, 102847092,
                         22325892, 46567936, 0, 0, 5849984])
        seqb = np.array([15218200, 22649542, 32968946, 34074908, 32432388, 33835304, 33005840,
                         31504828, 32986552, 33982704, 21082420, 0, 6310749, 10778815, 20097980,
                         15016198, 15131134, 20098040, 16748702, 10789100, 12638762, 10044920,
                         18176138, 11946176, 12542716, 17612744, 10046856, 10039904, 11782640,
                         18240248, 10149600, 10036736, 10036200, 12513772, 17482028, 10139604,
                         10031344, 15684360, 11825084, 12604436, 10023076, 20062748, 12512904,
                         17569756, 20054968, 20053980, 19213760, 15831984, 15086760, 19202840,
                         10858584, 19941208, 11843300, 18302928, 20034432, 19196348, 14018940,
                         19496324, 27381980, 30014520, 30010408, 20001320, 105505389, 66235531, 0,
                         0, 0, 5849984])
        seqc = np.array([15218308, 57413400, 74476256, 71961936, 74410664, 30261068, 0, 9403975,
                         7685589, 20097980, 20099320, 10048012, 20098040, 20089772, 19974352,
                         10157360, 20079146, 20100184, 20098444, 10046856, 20085840, 17501228,
                         12625324, 20072936, 10031472, 20070448, 14997736, 15098860, 20058104,
                         10024008, 20055592, 20060972, 20051920, 19954328, 20154620, 17477056,
                         22629156, 20049072, 22519492, 27566956, 20039632, 30050556, 30040060,
                         27461064, 27542672, 30008556, 32493672, 147570804, 26727808, 0, 0,
                         5849984])
        seqd = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        seqe = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.symbol_sequences = [sequence0, sequence1, sequence2, sequence3, sequence4,
                                 sequence5, sequence6, sequence7, sequence8, sequence9]

        self.labels = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9']
        self.dist_pairwise_mat = compute_distance_matrix(self.symbol_sequences,
                                                         type_dist='pairwise',
                                                         verbosity=0,
                                                         score_matrix=SCORE_MATRIX_FT)

        self.num_sequences = [seqa, seqb, seqc, seqd, seqe]

    def test_build_pw_distance_matrix(self):
        """ Build the matrix of distance (pairewise distance between each pair of sequence) """
        self.assertIsInstance(self.dist_pairwise_mat, np.ndarray)

    def test_build_ks_distance_matrix(self):
        """ Build the matrix of distance (ks distance between each pair of sequence) """
        dist_ks_delta_time_mat = compute_distance_matrix(self.num_sequences,
                                                         type_dist='KS-delta-time',
                                                         verbosity=0)
        self.assertIsInstance(dist_ks_delta_time_mat, np.ndarray)
        self.assertTrue((dist_ks_delta_time_mat.ravel() >= 0).all()
                        and (dist_ks_delta_time_mat.ravel() <= 1).all())
        dist_ks_pval_time_mat = compute_distance_matrix(self.num_sequences,
                                                        type_dist='KS-pvalue-time',
                                                        verbosity=0)
        self.assertIsInstance(dist_ks_pval_time_mat, np.ndarray)
        self.assertTrue((dist_ks_pval_time_mat.ravel() >= 0).all()
                        and (dist_ks_pval_time_mat.ravel() <= 1).all())
        dist_ks_delta_seq_mat = compute_distance_matrix(self.num_sequences,
                                                        type_dist='KS-delta-seq',
                                                        verbosity=0)
        self.assertIsInstance(dist_ks_delta_seq_mat, np.ndarray)
        self.assertTrue((dist_ks_delta_seq_mat.ravel() >= 0).all()
                        and (dist_ks_delta_seq_mat.ravel() <= 1).all())
        dist_ks_pval_seq_mat = compute_distance_matrix(self.num_sequences,
                                                       type_dist='KS-pvalue-seq',
                                                       verbosity=0)
        self.assertIsInstance(dist_ks_pval_seq_mat, np.ndarray)
        self.assertTrue((dist_ks_pval_seq_mat.ravel() >= 0).all()
                        and (dist_ks_pval_seq_mat.ravel() <= 1).all())

    def test_build_cvm_distance_matrix(self):
        """ Build the matrix of distance (cvm distance between each pair of sequence) """
        dist_cvm_delta_time_mat = compute_distance_matrix(self.num_sequences,
                                                          type_dist='CvM-delta-time',
                                                          verbosity=0)
        self.assertIsInstance(dist_cvm_delta_time_mat, np.ndarray)
        self.assertTrue((dist_cvm_delta_time_mat.ravel() >= 0).all())
        dist_cvm_pval_time_mat = compute_distance_matrix(self.num_sequences,
                                                         type_dist='CvM-pvalue-time',
                                                         verbosity=0)
        self.assertIsInstance(dist_cvm_pval_time_mat, np.ndarray)
        self.assertTrue((dist_cvm_pval_time_mat.ravel() >= 0).all()
                        and (dist_cvm_pval_time_mat.ravel() <= 1).all())
        dist_cvm_delta_seq_mat = compute_distance_matrix(self.num_sequences,
                                                         type_dist='CvM-delta-seq',
                                                         verbosity=0)
        self.assertIsInstance(dist_cvm_delta_seq_mat, np.ndarray)
        self.assertTrue((dist_cvm_delta_seq_mat.ravel() >= 0).all())
        dist_cvm_pval_seq_mat = compute_distance_matrix(self.num_sequences,
                                                        type_dist='CvM-pvalue-seq',
                                                        verbosity=0)
        self.assertIsInstance(dist_cvm_pval_seq_mat, np.ndarray)
        self.assertTrue((dist_cvm_pval_seq_mat.ravel() >= 0).all()
                        and (dist_cvm_pval_seq_mat.ravel() <= 1).all())

    def test_granularity_management(self):
        """ Build the matrix of distance using granularity 10 """
        dist_ks_pval_seq_mat = compute_distance_matrix(self.num_sequences,
                                                       type_dist='KS-pvalue-seq',
                                                       verbosity=0,
                                                       granularity=10)
        self.assertIsInstance(dist_ks_pval_seq_mat, np.ndarray)
        self.assertTrue((dist_ks_pval_seq_mat.ravel() >= 0).all()
                        and (dist_ks_pval_seq_mat.ravel() <= 1).all())

        with self.assertWarns(UserWarning) as warn_test:
            self.dist_pairwise_mat = compute_distance_matrix(self.symbol_sequences,
                                                             type_dist='pairwise',
                                                             verbosity=0,
                                                             granularity=10,
                                                             score_matrix=SCORE_MATRIX_FT)
        self.assertIn("Granularity", str(warn_test.warning))

    def test_build_distance_tree(self):
        """ Build the clustering tree (binary tree) from the matrix of distance scores """
        tree = build_distance_tree(self.dist_pairwise_mat, self.labels, diff_scale=10)
        self.assertIsInstance(tree, nj.distance_tree.Tree)


if __name__ == '__main__':
    unittest.main(verbosity=2)
