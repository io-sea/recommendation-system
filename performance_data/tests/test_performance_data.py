#!/usr/bin/env python

"""Tests for `performance_data` package."""


import os
from os.path import dirname
import unittest
import pandas as pd
from performance_data.performance_data import PhasePerformance


class TestPhasePerformance(unittest.TestCase):
    """Tests for `performance_data` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        current_dir = dirname(dirname(os.path.abspath(__file__)))
        print(current_dir)
        dataset_file = os.path.join(current_dir, "performance_data", "dataset",
                                    "performance_model_dataset_test.csv")
        self.dataset = pd.read_csv(dataset_file)
        self.phases = self.dataset.to_dict('records')
        self.targets = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")



    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_(self):
        """Test something."""
        print(self.phases)
        phases_perf = PhasePerformance(self.phases, self.targets, self.accelerator, self.ioi)

