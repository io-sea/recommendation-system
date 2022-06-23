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

from jobmodel import phase_sequencer
from jobmodel.phase_models import MODEL
from jobmodel.mongodb_extractor import MetricsDataBuilder
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


class TestSequencer(unittest.TestCase):
    """ TestCase used to test 'phase_sequencer' module."""
    def setUp(self):
        self.dict_db = dbc.load_collections(dbc.connect(HOST, PORT, NAME))
        self.phseq = phase_sequencer.PhaseSequencer(MODEL)
        self.jobid = 6012
        self.phseq.set_data(self.dict_db, self.jobid)

        self.order = ['Undefined', 'IOActive', 'Checkpoint', 'Start', 'End', 'IOInactive']
        self.sym_list = ['U', 'A', 'C', 'S', 'E', 'I']
        self.seq = self.phseq.get_phase_sequence(self.order, self.sym_list)

        self.mdb = {self.jobid: MetricsDataBuilder(self.dict_db, self.jobid)}
        self.all_metrics = {'FileIOSummaryGw': ['operationRead',
                                                'operationWrite'],
                            'IOSizesGw': ['read', 'write']}

    def test_load_db(self):
        """ loading the required database for testing (mongodb_extractor) """
        self.assertIsInstance(self.dict_db, dict)

    def test_init_phase_sequencer(self):
        """ Setting classification model into phases phase_sequencer """
        self.assertIsInstance(self.phseq.model, dict)

    def test_setup_phase_sequencer(self):
        """ Setting data into phases phase_sequencer """
        self.assertIsInstance(self.phseq.data, pd.DataFrame)

    def test_get_splited_model(self):
        """ Getting splited phases model for a given job """
        sequences = self.phseq.get_splited_phase_sequences()
        self.assertIsInstance(sequences, dict)

    def test_get_phase_sequence(self):
        """ Getting full phases sequence for a given job """
        self.assertIsInstance(self.seq, str)

    def test_phase_correction(self):
        """ Instantiate a phaseCorrector and generate a new corrected sequence """
        feature_mat = phase_sequencer.build_matrix(MetricsDataBuilder(self.dict_db,
                                                                      self.jobid))
        ph_corr = phase_sequencer.PhaseCorrector(self.seq)
        new_seq, ndiff = ph_corr.seq_verif(feature_mat, verbosity=1)
        cpt = 0
        while ndiff > 0 and cpt < 1000:
            ph_corr = phase_sequencer.PhaseCorrector(new_seq)
            new_seq, ndiff = ph_corr.seq_verif(feature_mat, verbosity=0)
            cpt += 1
        self.assertTrue(cpt < 1000) # divergence criteria
        self.assertIsInstance(new_seq, str)

    def test_get_metric_sequences(self):
        """ Getting IOI metric sequence for a list of job """
        sequences = phase_sequencer.get_metric_sequences(self.mdb, self.all_metrics)
        self.assertIsInstance(sequences, dict)


if __name__ == '__main__':
    unittest.main(verbosity=2)
