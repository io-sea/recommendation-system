#!/usr/bin/env python

"""Tests for `performance_data` package."""

import os
from os.path import dirname
import unittest
from abc import ABC, abstractmethod
import random
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from performance_data.data_table import PhaseData, DataTable
from performance_data.data_model import PhaseGenerator, AbstractModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

# whether or not populate the dataset file with new data
__POPULATE_DATASET__ = False


class TestPhaseGenerator(unittest.TestCase):
    """A unit test class for the PhaseGenerator class."""

    def setUp(self):
        """Create an instance of the DataGenerator class for use in the tests."""
        self.num_entries = 200
        self.volume = 50e6
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


@unittest.skipIf(not __POPULATE_DATASET__, "Test skipped to not populate the dataset file.")
class TestCompleteDataTable(unittest.TestCase):

    def setUp(self):
        self.target = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
        self.accelerator = False  # "SBB"
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
        # self.assertTrue(df.equals(complete_data))


class TestAbstractModel(unittest.TestCase):
    """
    A test suite for the AbstractModel class and its methods.
    """

    def setUp(self):
        """
        Sets up a DummyModel instance for testing.
        """
        class DummyModel(AbstractModel):
            def _create_model(self):
                return LinearRegression()
        self.model = DummyModel()

    def test_models_exist(self):
        """
        Sets up a DummyModel instance for testing.
        """
        self.assertIsInstance(self.model.model, dict)
        self.assertIsInstance(self.model.data, dict)

    def test_model_path(self):
        """
        Tests that the model_path attribute is a string.
        """
        self.assertIsInstance(self.model.model["nfs_bw"]["model_path"], str)
        self.assertIsInstance(self.model.model["lfs_bw"]["model_path"], str)

    def test_prepare_data(self):
        """
        Tests that the data, X, and y attributes are not None and are of the correct type.
        """
        self.assertIsNotNone(self.model.data)
        self.assertIsInstance(self.model.data["X_train"], pd.DataFrame)
        self.assertFalse(self.model.data["X_train"].empty)
        # print(type(self.model.data["nfs_bw"]["y_train"]))
        self.assertIsInstance(self.model.data["nfs_bw"]["y_train"], pd.Series)
        self.assertIsInstance(self.model.data["lfs_bw"]["y_train"], pd.Series)

    def test_prepare_input_data(self):
        """
        Tests that the _prepare_input_data method returns a DataFrame with the correct columns.
        """
        some_data = pd.DataFrame({'nodes': [1, 1, 1, 1],
                                  'read_volume': [20e6, 30e6, 30e6, 30e6],
                                  'write_volume': [10e6, 50e6, 30e6, 30e6],
                                  'read_io_pattern': ['rand', 'uncl', 'stride', 'seq'],
                                  'write_io_pattern': ['stride', 'seq', 'uncl', 'rand'], 'read_io_size': [512e3, 4e3, 8e6, 1e6],
                                  'write_io_size': [512e3, 8e6, 4e3, 1e6]})
        some_input_data = self.model._prepare_input_data(some_data)
        self.assertIsInstance(some_input_data, pd.DataFrame)
        self.assertLessEqual(len(some_data.columns), len(some_input_data.columns))

    def test_model_is_trained(self):
        """
        Tests that the model attribute is not None after training the model.
        """
        self.model.train_model()
        self.assertIsNotNone(self.model.model)

    def test_train_evaluate(self):
        """
        Test the training, evaluation, and prediction functionality of the model.

        Trains the model, evaluates it on test data, and makes predictions on new data to ensure that the model is functioning as expected. The test passes if the model is successfully trained and evaluated, and the predicted output has the expected shape.

        Returns:
            None.
        """
        self.model.train_model()
        score = self.model.evaluate_model()

    def test_predict(self):
        # Assert that the model has been trained and will predict on new data
        new_data = pd.DataFrame({'nodes': [1, 1, 1, 1],
                                 'read_volume': [20e6, 30e6, 30e6, 30e6],
                                 'write_volume': [10e6, 50e6, 30e6, 30e6],
                                 'read_io_pattern': ['rand', 'uncl', 'stride', 'seq'], 'write_io_pattern': ['stride', 'seq', 'uncl', 'rand'], 'read_io_size': [512e3, 4e3, 8e6, 1e6],
                                 'write_io_size': [512e3, 8e6, 4e3, 1e6]})
        prepared_new_data = self.model._prepare_input_data(new_data)
        self.assertIsInstance(prepared_new_data, pd.DataFrame)
        # train on initial data
        self.model.train_model()
        predictions = self.model.predict(new_data)
        self.assertIsInstance(predictions, dict)
        self.assertTrue(bool(predictions))
        for key, prediction in predictions.items():
            self.assertEqual(prediction.shape[0], 4)
            self.assertIsInstance(prediction, np.ndarray)


if __name__ == '__main__':
    unittest.main()
