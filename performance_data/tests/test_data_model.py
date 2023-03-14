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
from performance_data.data_model import PhaseGenerator, RegressionModel, TierModel, DataModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression

# whether or not populate the dataset file with new data
__POPULATE_DATASET__ = False


class TestPhaseGenerator(unittest.TestCase):
    """A unit test class for the PhaseGenerator class."""

    def setUp(self):
        """Create an instance of the DataGenerator class for use in the tests."""
        self.num_entries = 500
        self.volume = 500e6
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
        # self.target = dict(lfs="/fsiof/mimounis/tmp", nfs="/scratch/mimounis/tmp")
        # self.target = dict(lfs="/fsiof/mimounis/tmp", nfs="/home_nfs/mimounis/tmp")
        self.target = dict(lfs="/fsiof/mimounis/tmp", fs1="/fs1/mimounis/tmp")
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


class TestTierModel(unittest.TestCase):
    """Unit tests for the TierModel class."""

    def setUp(self):
        """Initializes the test data."""
        self.X, self.y = make_regression(n_samples=100, n_features=10, random_state=42)

    def test_fit_predict(self):
        """Tests that the TierModel can fit to data and make predictions."""
        model = TierModel()
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape, (100,))

    def test_fit_predict_with_custom_regressor(self):
        """Tests that the TierModel can fit to data and make predictions using a custom regressor."""
        from sklearn.tree import DecisionTreeRegressor
        model = TierModel(regressor=DecisionTreeRegressor())
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape, (100,))

    def test_fit_raises_error(self):
        """Tests that fitting the TierModel to invalid input data raises a ValueError."""
        model = TierModel()
        with self.assertRaises(ValueError):
            model.fit("not a valid input", self.y)

    def test_predict_raises_error(self):
        """Tests that attempting to predict with the TierModel using invalid input data raises a ValueError."""
        model = TierModel()
        with self.assertRaises(ValueError):
            model.predict("not a valid input")


class TestDataModel(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({'read_volume': [10, 20, 30],
                                  'write_volume': [20, 40, 60],
                                  'read_io_size': [100, 200, 300],
                                  'write_io_size': [50, 100, 150],
                                  'read_io_pattern': ['uncl', 'rand', 'seq'],
                                  'write_io_pattern': ['seq', 'stride', 'rand'],
                                  'col1_bw': [4, 5, 6],
                                  'col2_bw': [4, 5, 6]})
        self.model = DataModel()
        self.model.input_data = self.data

    @patch('pandas.read_csv')
    def test_init(self, mock_read_csv):
        # test default models parameter
        data_model = DataModel()
        self.assertIsInstance(data_model.models, list)
        self.assertEqual(len(data_model.models), 0)

        # test non-default models parameter
        mock_model = TierModel()
        data_model = DataModel(models=[mock_model])
        self.assertEqual(len(data_model.models), 1)
        self.assertIsInstance(data_model.models[0], TierModel)

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        # test successful loading of data
        mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2, 3], 'col2_bw': [4, 5, 6]})
        data_model = DataModel()
        data_model.load_data()
        self.assertIsInstance(data_model.input_data, pd.DataFrame)
        self.assertListEqual(data_model.target_tiers, ["col2_bw"])

        # test loading empty data
        mock_read_csv.return_value = pd.DataFrame()
        data_model = DataModel()
        with self.assertRaises(AssertionError):
            data_model.load_data()

    def test_prepare_input_data(self):
        preprocessed_data = self.model._prepare_input_data(self.data)
        self.assertEqual(preprocessed_data.shape, (3, 13))
        self.assertCountEqual(preprocessed_data.columns, ['cat__read_io_pattern_rand',
                                                          'cat__read_io_pattern_stride',
                                                          'cat__read_io_pattern_seq', 'cat__read_io_pattern_uncl',
                                                          'cat__write_io_pattern_rand', 'cat__write_io_pattern_stride',
                                                          'cat__write_io_pattern_seq', 'cat__write_io_pattern_uncl',
                                                          'remainder__read_io_size', 'remainder__write_io_size',
                                                          'remainder__read_ratio', 'remainder__write_ratio',
                                                          'remainder__avg_io_size'])

    def test_prepare_data(self):
        X, y = self.model._prepare_data()
        print(y)
        expected_y = pd.DataFrame({
            'col1_bw': [4.8e5, 3e5, 2.4e5],
            'col2_bw': [4.8e5, 3e5, 2.4e5]
        })
        self.assertListEqual(list(expected_y["col1_bw"]),  [4.8e5, 3e5, 2.4e5])
        self.assertListEqual(list(expected_y["col2_bw"]),  [4.8e5, 3e5, 2.4e5])

class TestDataModelTraining(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
                              columns=['col_1', 'col_2', 'col_3'])
        self.y = pd.DataFrame(np.array([[10, 20], [40, 50], [70, 80], [100, 110]]),
                              columns=['target_1_bw', 'target_2_bw'])
        self.data_model = DataModel(models={})

    def test_train_models(self):
        model_1 = TierModel(regressor=LinearRegression())
        model_2 = TierModel(regressor=LinearRegression())
        models = {'target_1_bw': model_1, 'target_2_bw': model_2}
        self.data_model.models = models
        trained_models = self.data_model.train_model(X=self.X, y=self.y)
        self.assertEqual(len(trained_models), len(models))
        for key in models.keys():
            self.assertIn(key, trained_models)
            self.assertIsNotNone(trained_models[key])
            self.assertIsInstance(trained_models[key], TierModel)
            self.assertIsNot(trained_models[key], models[key])


            # Add additional checks here to ensure that the trained models are correct.
# class TestAbstractModel(unittest.TestCase):
#     """
#     A test suite for the AbstractModel class and its methods.
#     """

#     def setUp(self):
#         """
#         Sets up a DummyModel instance for testing.
#         """
#         class DummyModel(AbstractModel):
#             def _create_model(self):
#                 return LinearRegression()
#         self.model = DummyModel()

#     def test_models_exist(self):
#         """
#         Sets up a DummyModel instance for testing.
#         """
#         self.assertIsInstance(self.model.model, dict)
#         self.assertIsInstance(self.model.data, dict)

#     def test_model_path(self):
#         """
#         Tests that the model_path attribute is a string.
#         """
#         for tier in self.model.target_tiers:
#             self.assertIsInstance(self.model.model[tier]["model_path"], str)

#     def test_prepare_data(self):
#         """
#         Tests that the data, X, and y attributes are not None and are of the correct type.
#         """
#         self.assertIsNotNone(self.model.data)
#         self.assertIsInstance(self.model.data["X_train"], pd.DataFrame)
#         self.assertFalse(self.model.data["X_train"].empty)
#         for tier in self.model.target_tiers:
#             self.assertIsInstance(self.model.data[tier]["y_train"], pd.Series)

#     def test_prepare_input_data(self):
#         """
#         Tests that the _prepare_input_data method returns a DataFrame with the correct columns.
#         """
#         some_data = pd.DataFrame({'nodes': [1, 1, 1, 1],
#                                   'read_volume': [20e6, 30e6, 30e6, 30e6],
#                                   'write_volume': [10e6, 50e6, 30e6, 30e6],
#                                   'read_io_pattern': ['rand', 'uncl', 'stride', 'seq'],
#                                   'write_io_pattern': ['stride', 'seq', 'uncl', 'rand'], 'read_io_size': [512e3, 4e3, 8e6, 1e6],
#                                   'write_io_size': [512e3, 8e6, 4e3, 1e6]})
#         some_input_data = self.model._prepare_input_data(some_data)
#         self.assertIsInstance(some_input_data, pd.DataFrame)
#         self.assertLessEqual(len(some_data.columns), len(some_input_data.columns))

#     @patch('joblib.dump')
#     def test_save_model(self, mock_dump):
#         # Set up a mock model path
#         model_path = 'path/to/model.joblib'

#         # Call the save_model method
#         self.model.save_model()

#         # Check that dump was called
#         self.assertTrue(mock_dump.called)

#     def test_model_is_trained(self):
#         """
#         Tests that the model attribute is not None after training the model.
#         """
#         self.model.train_model()
#         self.assertIsNotNone(self.model.model)

#     def test_train_evaluate(self):
#         """
#         Test the training, evaluation, and prediction functionality of the model.

#         Trains the model, evaluates it on test data, and makes predictions on new data to ensure that the model is functioning as expected. The test passes if the model is successfully trained and evaluated, and the predicted output has the expected shape.

#         Returns:
#             None.
#         """
#         self.model.train_model()
#         score = self.model.evaluate_model()

#     def test_predict(self):
#         # Assert that the model has been trained and will predict on new data
#         new_data = pd.DataFrame({'nodes': [1, 1, 1, 1],
#                                  'read_volume': [20e6, 30e6, 30e6, 30e6],
#                                  'write_volume': [10e6, 50e6, 30e6, 30e6],
#                                  'read_io_pattern': ['rand', 'uncl', 'stride', 'seq'], 'write_io_pattern': ['stride', 'seq', 'uncl', 'rand'], 'read_io_size': [512e3, 4e3, 8e6, 1e6],
#                                  'write_io_size': [512e3, 8e6, 4e3, 1e6]})
#         prepared_new_data = self.model._prepare_input_data(new_data)
#         self.assertIsInstance(prepared_new_data, pd.DataFrame)
#         # train on initial data
#         self.model.train_model()
#         predictions = self.model.predict(new_data)
#         self.assertIsInstance(predictions, dict)
#         self.assertTrue(bool(predictions))
#         for key, prediction in predictions.items():
#             self.assertEqual(prediction.shape[0], 4)
#             self.assertIsInstance(prediction, np.ndarray)


# class TestLoadModel(unittest.TestCase):
#     def setUp(self):
#         """
#         Sets up a DummyModel instance for testing.
#         """
#         class DummyModel(AbstractModel):
#             def _create_model(self):
#                 return LinearRegression()
#         self.model = DummyModel()
#         self.model.train_model()
#         self.model.save_model()
#         self.new_data = pd.DataFrame({'nodes': [1, 1, 1, 1],
#                                       'read_volume': [20e6, 30e6, 30e6, 30e6],
#                                       'write_volume': [10e6, 50e6, 30e6, 30e6],
#                                       'read_io_pattern': ['rand', 'uncl', 'stride', 'seq'], 'write_io_pattern': ['stride', 'seq', 'uncl', 'rand'], 'read_io_size': [512e3, 4e3, 8e6, 1e6],
#                                       'write_io_size': [512e3, 8e6, 4e3, 1e6]})
#     def test_load(self):
#         # NOTE: to be finished
#         for tier in self.model.target_tiers:
#             path = self.model.model[tier]["model_path"]
#         loaded_model = load_model(path)
#         print(loaded_model)
#         print(type(loaded_model))
#         predictions = loaded_model["model"].predict(self.new_data)
#         print(predictions)
if __name__ == '__main__':
    unittest.main()
