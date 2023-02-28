#!/usr/bin/env python

"""Tests for `performance_data` package."""


import os
from os.path import dirname
import unittest
import random
import pandas as pd
from unittest.mock import MagicMock, patch
from performance_data.data_table import PhaseData, DataTable
from performance_data.data_model import PhaseGenerator
from performance_data import data_table

class TestPhaseData(unittest.TestCase):
    """Tests for the `PhaseData` class."""
    def setUp(self):
        """Set up test fixtures."""
        self.phases = [dict(read_volume=1e8, read_io_pattern="stride", read_io_size=1e4,
                          write_volume=1e8, write_io_pattern="uncl", write_io_size=1e4,
                          nodes=1),
                       dict(read_volume=1e8, read_io_pattern="stride", read_io_size=1e4,
                          write_volume=1e8, write_io_pattern="seq", write_io_size=1e4,
                          nodes=1)]

        self.targets = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
        self.ioi = {"nfs": "/nfs", "lfs": "/lfs"}
        self.sample = 2
        self.lite = True
        self.phase_data = PhaseData(self.phases, self.targets, self.ioi, self.sample, self.lite)

    def test_get_phase_data(self):
        """Test the `get_phase_data` method of the `PhaseData` class.

        The test creates a `PhaseData` instance with a given set of parameters and a mocked implementation of `run_phase_workload`, then calls the `get_phase_data` method with various sets of target tiers, and compares the results with the expected dataframes.

        Returns:
            None
        """
        self.phase_data.run_phase_workload = MagicMock(return_value=42)
        expected_df = pd.DataFrame({
            "nfs_bw": [42, 42],
            "lfs_bw": [42, 42],
            "sbb_bw": [42, 42]
        })

        # Test performance calculation on all tiers
        result_df = self.phase_data.get_phase_data()
        pd.testing.assert_frame_equal(result_df, expected_df)

        # Test performance calculation on a subset of tiers
        result_df = self.phase_data.get_phase_data(target_names=["nfs_bw", "sbb_bw"])
        expected_df = pd.DataFrame({
            "nfs_bw": [42, 42],
            "sbb_bw": [42, 42]
        })
        pd.testing.assert_frame_equal(result_df, expected_df)

    def tearDown(self):
        """Tear down test fixtures."""


class TestDataTable(unittest.TestCase):
    """Tests for `performance_data` package."""
    def setUp(self):
        """Set up test fixtures, if any."""
        current_dir = dirname(dirname(os.path.abspath(__file__)))
        self.filename = os.path.join(current_dir, "tests",
                                         "test_data", "test_dataset.csv")
        self.output_filename = os.path.join(current_dir, "tests",
                                         "test_data", "test_dataset_complete.csv")
        self.targets = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
        self.accelerator = False
        self.sample = 2
        self.ioi = False
        self.lite = False
        self.data_table = DataTable(self.targets, self.accelerator, self.sample, self.ioi,
                                    filename = self.filename, lite=self.lite)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        try:
            print(f"content of {self.output_filename}:\n {pd.read_csv(self.output_filename)}")
            print("cleaning output file...")
            os.remove(self.output_filename)

        except OSError:
            pass

    def test_get_tiers(self):
        """Test if get_tiers_from_targets method returns a list of tiers extracted from the target directory and accelerator attribute."""
        self.assertListEqual(self.data_table.get_tiers_from_targets(), ["lfs_bw", "nfs_bw"])

    @patch.object(PhaseData, 'get_phase_data')
    def test_get_performance_table(self, mock_phase_data):
        """Test if get_performance_table method correctly returns performance data in a dataframe with expected values."""
        mock_phase_data.return_value = pd.DataFrame({"lfs_bw": [42]*8, "nfs_bw": [42]*8})
        perf_data = self.data_table.get_performance_table(output_filename=self.output_filename)
        self.assertListEqual(list(perf_data[["lfs_bw"]].to_numpy().flatten()), [42]*8)
        self.assertListEqual(list(perf_data[["nfs_bw"]].to_numpy().flatten()), [42]*8)


