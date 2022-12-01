#!/usr/bin/env python

"""Tests for `performance_data` package."""


import os
from os.path import dirname
import unittest
import pandas as pd
from performance_data.performance_data import PhaseData, DataTable


class TestPhaseData(unittest.TestCase):
    """Tests for `performance_data` package."""
    def setUp(self):
        """Set up test fixtures, if any."""
        current_dir = dirname(dirname(os.path.abspath(__file__)))
        print(current_dir)
        self.dataset_file = os.path.join(current_dir, "performance_data", "dataset",
                                    "performance_model_dataset_test.csv")
        self.dataset = pd.read_csv(self.dataset_file)
        self.phases = self.dataset.to_dict('records')
        self.targets = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
        self.ioi = False

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_(self):
        """Test something."""
        #print(self.phases)
        phases_data = PhaseData(self.phases, self.targets, self.ioi)
        df = phases_data.get_phase_data()
        #print(df)
        dt = DataTable(self.targets, accelerator=True, filename=self.dataset_file)
        df = dt.get_performance_table()
        #print(df)

    def test_standalone(self):
        """Tests the whole pipeline."""
        targets = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
        dt = DataTable(targets, accelerator=True, filename=self.dataset_file)
        df = dt.get_performance_table()



