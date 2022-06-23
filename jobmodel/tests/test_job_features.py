""" IOPA jobmodel unittests module """

from __future__ import division, absolute_import, generators, print_function, unicode_literals, \
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2017-2019 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import pickle
import unittest
from unittest.mock import MagicMock
from os import path
import pandas as pd
import numpy as np

from jobmodel import job_features
from ioanalyticstools import dbconnector as dbc


HOST = None
PORT = None
NAME = None
# Load the minimal db serialized in a pickle file
PATH_MOCK_DB = path.dirname(path.abspath(__file__)) + '/database'
with open(PATH_MOCK_DB + '/mock_db.pkl', 'rb') as mock_db:
    DB = pickle.load(mock_db)
dbc.connect = MagicMock()
dbc.load_collections = MagicMock(return_value=DB)


class TestFeatures(unittest.TestCase):
    """ TestCase used to test 'job_features' module."""
    def setUp(self):
        seq0 = 'SSSSSSSUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCCCUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCCCUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UCCCCUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCCCUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUEEE'

        seq1 = 'SSSSSSSUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUCCCCCUCCCCUUUUUUUUUUUUUUUUUUUUUUCCCUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCCCCUUCCCCCUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
CCCCCUCCCCUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCCCCUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCCCCCCCUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCCCUCCCUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\
UUUUUUUUUUUUUUUUUUUEEE'

        seq2 = 'CCXCUUUUUUUUUUUUCCCUUUUUUUUUUUCCXXCCUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCCXXXCCC'
        seq3 = 'CCCSSUUUUCCCUUCUUUCCUCUUEEC'
        self.seq = [seq0, seq1, seq2, seq3]

    def test_phase_boundaries_tol_default(self):
        """Get the boundary of each targeted phases found in sequences, with tol=0"""
        ref_bounds = [[(225, 227), (447, 449), (671, 674), (896, 898)],
                      [(201, 205), (207, 210), (233, 235), (333, 336), (339, 343), (464, 468),
                       (470, 473), (694, 697), (784, 790), (926, 928), (930, 932)],
                      [(0, 1), (3, 3), (16, 18), (30, 31), (34, 35), (68, 69), (73, 75)],
                      [(0, 2), (9, 11), (14, 14), (18, 19), (21, 21), (26, 26)]]
        for i, seq in enumerate(self.seq):
            bounds = job_features.get_phase_boundaries(seq)
            self.assertIsInstance(bounds, list)
            self.assertListEqual(bounds, ref_bounds[i])

    def test_phase_boundaries_tol_1(self):
        """Get the boundary of each targeted phases found in sequences, with tol=1"""
        ref_bounds = [[(225, 227), (447, 449), (671, 674), (896, 898)],
                      [(201, 210), (233, 235), (333, 336), (339, 343), (464, 473), (694, 697),
                       (784, 790), (926, 932)],
                      [(0, 3), (16, 18), (30, 31), (34, 35), (68, 69), (73, 75)],
                      [(0, 2), (9, 11), (14, 14), (18, 21), (26, 26)]]
        for i, seq in enumerate(self.seq):
            bounds = job_features.get_phase_boundaries(seq, tol=1)
            self.assertIsInstance(bounds, list)
            self.assertListEqual(bounds, ref_bounds[i])

    def test_phase_boundaries_tol_3(self):
        """Get the boundary of each targeted phases found in sequences, with tol=3"""
        ref_bounds = [[(225, 227), (447, 449), (671, 674), (896, 898)],
                      [(201, 210), (233, 235), (333, 343), (464, 473), (694, 697), (784, 790),
                       (926, 932)],
                      [(0, 3), (16, 18), (30, 35), (68, 75)],
                      [(0, 2), (9, 21), (26, 26)]]
        for i, seq in enumerate(self.seq):
            bounds = job_features.get_phase_boundaries(seq, tol=3)
            self.assertIsInstance(bounds, list)
            self.assertListEqual(bounds, ref_bounds[i])

    def test_phase_boundaries_tol_7(self):
        """Get the boundary of each targeted phases found in sequences, with tol=3"""
        ref_bounds = [[(225, 227), (447, 449), (671, 674), (896, 898)],
                      [(201, 210), (233, 235), (333, 343), (464, 473), (694, 697), (784, 790),
                       (926, 932)],
                      [(0, 3), (16, 18), (30, 35), (68, 75)],
                      [(0, 26)]]
        for i, seq in enumerate(self.seq):
            bounds = job_features.get_phase_boundaries(seq, tol=7)
            self.assertIsInstance(bounds, list)
            self.assertListEqual(bounds, ref_bounds[i])

    def test_extract_phase_only_C(self):
        """Get the phases of the targeted phases found in sequences, with tol=0"""
        seq = 'CCCCCCCCCCCC'
        ref_phases = np.array([True, True, True, True, True, True, True, True, True, True, True,
                               True])
        phases = job_features.extract_phases(seq)
        np.testing.assert_array_equal(phases, ref_phases)

    def test_extract_phase_no_C(self):
        """Get the phases of the targeted phases found in sequences, with tol=0"""
        seq = ''
        ref_phases = np.array([])
        phases = job_features.extract_phases(seq, tol=2)
        np.testing.assert_array_equal(phases, ref_phases)


if __name__ == '__main__':
    unittest.main(verbosity=2)
