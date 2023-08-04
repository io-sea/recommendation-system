#!/usr/bin/env python

"""Tests for `performance_data` on deep platform."""

import os
import sys
from os.path import dirname
import unittest
from abc import ABC, abstractmethod
import random
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from performance_data.data_table import PhaseData, DataTable
from performance_data.data_model import PhaseGenerator, RegressionModel, TierModel, DataModel, DeepNNModel
from performance_data.data_model import load_and_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from loguru import logger


# whether or not populate the dataset file with new data
__POPULATE_DATASET__ = True
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DATA = os.path.join(os.path.dirname(CURRENT_DIR), "tests", "test_data",
                         "test_deep_dataset.csv")
TEST_MODELS = os.path.join(os.path.dirname(CURRENT_DIR), "tests", "test_data",
                           "test_deep_models")


class TestPhasesTable(unittest.TestCase):
    """A unit test class for the PhaseGenerator class."""

    def setUp(self):
        """Create an instance of the DataGenerator class for use in the tests."""
        self.num_entries = 3
        self.volume = 50e6
        self.generator = PhaseGenerator(num_entries=self.num_entries,
                                        volume=self.volume)
        current_dir = dirname(dirname(os.path.abspath(__file__)))
        self.filename = os.path.join(current_dir, "tests",
                                     "deep_data", "deep_small_dataset.csv")

    def test_generate_data(self):
        """Test the generate_data method of the DataGenerator class.

        Test that the method returns a list of dictionaries with the correct keys,
        and that the values for each key are within the expected bounds.
        """
        data = self.generator.generate_data()

        self.assertEqual(len(data), self.num_entries)
        for entry in data:
            self.assertIn(entry["read_io_pattern"], ["uncl", "seq", "rand", "stride"])
            self.assertIn(entry["write_io_pattern"], ["uncl", "seq", "rand", "stride"])
            self.assertIn(entry["read_io_size"], [4e3, 16e3, 128e3, 512e3, 2e6, 8e6])
            self.assertIn(entry["write_io_size"], [4e3, 16e3, 128e3, 512e3, 2e6, 8e6])
            self.assertGreaterEqual(entry["read_volume"], 0.0)
            self.assertLess(entry["read_volume"], self.volume)
            self.assertGreaterEqual(entry["write_volume"], 0.0)
            self.assertLess(entry["write_volume"], self.volume)

    def test_export_data(self):
        """Test the export_data method of the PhaseGenerator class.

        Test that the method exports the generated data to a CSV file with the
        specified name, and that the exported file can be read and contains the
        expected data.
        """
        self.generator.export_data(self.filename)
        print(self.filename)
        self.assertTrue(os.path.exists(self.filename))
        df = pd.read_csv(self.filename)
        self.assertEqual(df.shape[0], self.num_entries)
        self.assertEqual(df.shape[1], 7)
        self.assertIn("read_io_pattern", df.columns)
        self.assertIn("write_io_pattern", df.columns)
        self.assertIn("read_io_size", df.columns)
        self.assertIn("write_io_size", df.columns)
        self.assertIn("read_volume", df.columns)
        self.assertIn("write_volume", df.columns)

        
class TestDataTable(unittest.TestCase):
    """Tests for `performance_data` package."""
    def setUp(self):
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        """Set up test fixtures, if any."""
        current_dir = dirname(dirname(os.path.abspath(__file__)))
        self.filename = os.path.join(current_dir, "tests",
                                     "deep_data", "test_deep_generated_dataset.csv")
        
        # self.output_filename = os.path.join(current_dir, "tests",
        #                                  "test_data", "test_deep_small_dataset_complete.csv")
        self.targets = dict(gpfs_nfs="/p/home/jusers/mimouni1/deep/recsys/iosea-wp3-recommandation-system/performance_data/performance_data/tmp",
                            afsm_beegfs="/afsm/iosea/mimouni1/perf_data")
                            #beegfs_old="/work_old/iosea/mimouni1")
        self.accelerator = True
        self.sample = 1
        self.ioi = False
        self.lite = False
        self.data_table = DataTable(self.targets, 
                                    self.accelerator, 
                                    ioi=self.ioi,
                                    sample=self.sample, 
                                    filename=self.filename, 
                                    lite=self.lite)

    # def tearDown(self):
    #     """Tear down test fixtures, if any."""
    #     try:
    #         print(f"content of {self.output_filename}:\n {pd.read_csv(self.output_filename)}")
    #         print("cleaning output file...")
    #         os.remove(self.output_filename)

    #     except OSError:
    #         pass

    def test_get_tiers(self):
        """Test if get_tiers_from_targets method returns a list of tiers extracted from the target directory and accelerator attribute."""
        logger.info(f"targets: {self.data_table.get_tiers_from_targets()}")
        #self.assertListEqual(self.data_table.get_tiers_from_targets(), ["nfs_bw"])

    def test_get_performance_table(self):
        """Test if get_performance_table method correctly returns performance data in a dataframe with expected values."""
        perf_data = self.data_table.get_performance_table()
            #output_filename=self.output_filename)
        #print(perf_data)