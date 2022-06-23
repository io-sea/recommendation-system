""" IOPA jobmodel unittests module """

from __future__ import division, absolute_import, generators, print_function, unicode_literals, \
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2017 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import pickle
import unittest
from unittest.mock import MagicMock
from os import path
import pandas as pd

from jobmodel import mongodb_extractor
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


class TestDataBuilder(unittest.TestCase):
    """ TestCase used to test 'mongodb_extractor' module."""
    def setUp(self):
        # Loading mongo database as pandas dict
        self.dict_db = dbc.load_collections(dbc.connect(HOST, PORT, NAME))
        self.jobid = 6012
        # Build the matrix of normalized features for the job
        self.metrics_data = mongodb_extractor.MetricsDataBuilder(self.dict_db, self.jobid)

        self.collections = ["IODurationsGw", "IOSizesGw"]
        self.modes = ["read", "write"]

    def test_hist_col_zscore(self):
        """ collecting histograms using zscore normalisation"""
        hist = self.metrics_data.collect_histograms_from_db(self.collections,
                                                            self.modes,
                                                            norm="zscore")
        self.assertIsInstance(hist, pd.DataFrame)

    def test_hist_col_minmax(self):
        """ collecting histograms using minmax normalisation"""
        hist = self.metrics_data.collect_histograms_from_db(self.collections,
                                                            self.modes,
                                                            norm="minmax")
        self.assertIsInstance(hist, pd.DataFrame)

    def test_building_job_histograms(self):
        """ Build a dataframe of all histograms """
        hist = self.metrics_data.get_job_histograms()
        self.assertIsInstance(hist, pd.DataFrame)


if __name__ == '__main__':
    unittest.main(verbosity=2)
