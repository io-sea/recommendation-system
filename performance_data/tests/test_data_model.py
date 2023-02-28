#!/usr/bin/env python

"""Tests for `performance_data` package."""

import os
from os.path import dirname
import unittest
import random
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from performance_data.data_table import PhaseData, DataTable
from performance_data.data_model import PhaseGenerator, RegressionModel, RandomForestModel
from sklearn.model_selection import train_test_split

class TestPhaseGenerator(unittest.TestCase):
    """A unit test class for the PhaseGenerator class."""

    def setUp(self):
        """Create an instance of the DataGenerator class for use in the tests."""
        self.num_entries = 50
        self.volume = 100e6
        self.generator = PhaseGenerator(num_entries=self.num_entries,
                                       volume=self.volume)
        current_dir = dirname(dirname(os.path.abspath(__file__)))
        self.filename = os.path.join(current_dir, "tests",
                                     "test_data", "test_generated_dataset.csv")

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

class TestCompleteDataTable(unittest.TestCase):

    def setUp(self):
        self.target = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
        self.accelerator = "SBB"
        current_dir = dirname(dirname(os.path.abspath(__file__)))
        self.filename = os.path.join(current_dir, "tests",
                                     "test_data", "test_generated_dataset.csv")
        self.complete_filename = os.path.join(current_dir, "tests",
                                     "test_data", "complete_test_generated_dataset.csv")
        self.data_table = DataTable(self.target, accelerator=self.accelerator, filename=self.filename)

    def test_get_performance_table(self):
        df = self.data_table.get_performance_table(output_filename=self.complete_filename)
        self.assertIsNotNone(df)
        self.assertTrue(os.path.exists(self.complete_filename))

        # Check if the data in the complete file is the same as the one returned by the method
        complete_data = pd.read_csv(self.complete_filename)
        print(df)
        print(complete_data)
        self.assertTrue(df.equals(complete_data))



class TestRegressionModel(unittest.TestCase):
    """
    Unit tests for the RegressionModel class.
    """
    def setUp(self):
        """
        Initializes the test data for the regression model.
        """
        current_dir = dirname(dirname(os.path.abspath(__file__)))
        self.data_file = os.path.join(current_dir, "tests",
                                              "test_data", "complete_test_generated_dataset.csv")

        self.model = RegressionModel(self.data_file)
        self.model.train_model()
        self.new_data = self.model.X_test.iloc[0:1]
        self.predictions = self.model.predict(self.new_data)

    def test_train_model(self):
        """
        Tests if the model is correctly trained on the data.
        """
        self.assertIsNotNone(self.model.model)

    def test_evaluate_model(self):
        """
        Tests if the model score is correctly computed.
        """
        score = self.model.evaluate_model()
        print(score)
        self.assertIsInstance(score, float)

    def test_predict(self):
        """
        Tests if the predictions made by the model are valid.
        """
        self.assertIsInstance(self.predictions, np.ndarray)
        self.assertEqual(self.predictions.shape[0], self.new_data.shape[0])
        self.assertEqual(self.predictions.shape[1], self.model.y_train.shape[1])


class TestRandomForestModel(unittest.TestCase):

    def setUp(self):
        current_dir = dirname(dirname(os.path.abspath(__file__)))
        self.file_path = os.path.join(current_dir, "tests",
                                              "test_data", "complete_test_generated_dataset.csv")
        self.model = RandomForestModel(self.file_path, num_trees=10, max_depth=5)
        self.model.fit()

    def tearDown(self):
        del self.model

    def test_file_path(self):
        self.assertEqual(self.model.file_path, self.file_path)

    def test_num_trees(self):
        self.assertEqual(self.model.num_trees, 10)

    def test_max_depth(self):
        self.assertEqual(self.model.max_depth, 5)

    def test_X_train(self):
        self.assertIsInstance(self.model.X_train, pd.DataFrame)
        self.assertTrue(all(col in self.model.X_train.columns for col in ['nodes', 'read_volume', 'write_volume', 'read_io_size', 'write_io_size']))

    def test_X_test(self):
        self.assertIsInstance(self.model.X_test, pd.DataFrame)
        self.assertTrue(all(col in self.model.X_test.columns for col in ['nodes', 'read_volume', 'write_volume', 'read_io_size', 'write_io_size']))

    def test_y_train(self):
        self.assertIsInstance(self.model.y_train, pd.DataFrame)
        self.assertTrue(all(col in self.model.y_train.columns for col in ['lfs_bw', 'nfs_bw', 'sbb_bw']))

    def test_y_test(self):
        self.assertIsInstance(self.model.y_test, pd.DataFrame)
        self.assertTrue(all(col in self.model.y_test.columns for col in ['lfs_bw', 'nfs_bw', 'sbb_bw']))

    def test_predict(self):
        y_pred = self.model.predict()
        self.assertIsInstance(y_pred, np.ndarray)

    def test_evaluate(self):
        mse, r2 = self.model.evaluate()
        self.assertIsInstance(mse, float)
        self.assertIsInstance(r2, float)

if __name__ == '__main__':
    unittest.main()
