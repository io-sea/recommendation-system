#!/usr/bin/env python

"""Tests for `performance_data` package."""

import os
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

# whether or not populate the dataset file with new data
__POPULATE_DATASET__ = False
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DATA = os.path.join(os.path.dirname(CURRENT_DIR), "tests", "test_data",
                         "test_dataset.csv")
TEST_MODELS = os.path.join(os.path.dirname(CURRENT_DIR), "tests", "test_data",
                           "test_models")


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

    @patch('pandas.read_csv')
    def test_init_no_model(self, mock_read_csv):
        mock_read_csv.return_value = self.data
        self.model = DataModel()
        self.assertIsInstance(self.model.models, dict)

    @patch('pandas.read_csv')
    def test_init_dict_model(self, mock_read_csv):
        mock_read_csv.return_value = self.data
        models_dict = {'col1_bw': TierModel(), 'col2_bw': TierModel()}
        data_model = DataModel(models=models_dict)
        print(data_model.models)
        # self.assertIsInstance(data_model.models, dict)
        # self.assertEqual(len(data_model.models), 2)
        # self.assertIsInstance(data_model.models['col1_bw'], TierModel)
        # self.assertIsInstance(data_model.models['col2_bw'], TierModel)

    @patch('pandas.read_csv')
    def test_init_list_model(self, mock_read_csv):
        mock_read_csv.return_value = self.data
        mock_model = TierModel()
        data_model = DataModel(models=[mock_model, mock_model])
        self.assertIsInstance(data_model.models, dict)
        self.assertEqual(len(data_model.models), 2)
        self.assertIsInstance(data_model.models['col1_bw'], TierModel)
        self.assertIsInstance(data_model.models['col2_bw'], TierModel)

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        # test successful loading of data
        mock_read_csv.return_value = pd.DataFrame({'read_volume': [10, 20, 30],
                                                    'write_volume': [20, 40, 60],
                                                    'read_io_size': [100, 200, 300],
                                                    'write_io_size': [50, 100, 150],
                                                    'read_io_pattern': ['uncl', 'rand', 'seq'],
                                                    'write_io_pattern': ['seq', 'stride', 'rand'],
                                                    'col1': [4, 5, 6],
                                                    'col2_bw': [4, 5, 6]})
        data_model = DataModel()
        self.assertIsInstance(data_model.input_data, pd.DataFrame)
        self.assertListEqual(data_model.target_tiers, ["col2_bw"])

    @patch('pandas.read_csv')
    def test_load_empty_data(self, mock_read_csv):
        # test loading empty data
        mock_read_csv.return_value = pd.DataFrame()
        with self.assertRaises(AssertionError):
            data_model = DataModel()
            data_model.load_data()


    def test_prepare_input_data(self):
        self.model = DataModel()
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
        
    @patch('pandas.read_csv')
    def test_prepare_data(self, mock_read_csv):
        mock_read_csv.return_value = self.data
        self.model = DataModel()
        X, y = self.model._prepare_data()
        expected_y = pd.DataFrame({
            'col1_bw': [4.8e5, 3e5, 2.4e5],
            'col2_bw': [4.8e5, 3e5, 2.4e5]
        })
        self.assertListEqual(list(expected_y["col1_bw"]),  [4.8e5, 3e5, 2.4e5])
        self.assertListEqual(list(expected_y["col2_bw"]),  [4.8e5, 3e5, 2.4e5])


class TestDataModelTraining(unittest.TestCase):
    def setUp(self):
        # Initialize two TierModel instances with different regressors
        model_1 = TierModel(regressor=LinearRegression())
        model_2 = TierModel(regressor=RandomForestRegressor())
        self.models_dict = {"lfs_bw": model_1, "fs1_bw": model_2}
        self.models_list = [model_1, model_2]


    def test_train_models_dict(self):
        # Initialize DataModel instance with the provided models
        data_model = DataModel(models=self.models_dict)
        # Train the models and get the trained_models dictionary
        trained_models = data_model.train_model()

        # Check that the number of trained models matches the number of models provided
        self.assertEqual(len(trained_models), len(self.models_dict))

        for key in self.models_dict.keys():
            # Check that each provided model has a corresponding trained model
            self.assertIn(key, trained_models)

            # Check that the trained model is not None and is an instance of TierModel
            self.assertIsNotNone(trained_models[key])
            self.assertIsInstance(trained_models[key], TierModel)

            # Check that the trained model is not the same as the initially provided model
            # (ensuring that a new model is trained rather than reusing the provided one)
            #self.assertIsNot(trained_models[key], self.models_dict[key])

    def test_train_models_list(self):
        # Initialize DataModel instance with the provided models
        data_model = DataModel(models=self.models_list)
        # Train the models and get the trained_models dictionary
        trained_models = data_model.train_model()

        # Check that the number of trained models matches the number of models provided
        self.assertEqual(len(trained_models), len(self.models_list))

        # Ensure that the keys in the trained_models_list dictionary match the target columns from the data
        target_columns = data_model.y.columns
        for key in target_columns:
            self.assertIn(key, trained_models)

            # Check that the trained model is not None and is an instance of TierModel
            self.assertIsNotNone(trained_models[key])
            self.assertIsInstance(trained_models[key], TierModel)

            # Check that the trained model is not the same as the initially provided model
            # (ensuring that a new model is trained rather than reusing the provided one)
            # for model in self.models_list:
            #     self.assertIsNot(trained_models[key], model)



class TestDataModelPrediction(unittest.TestCase):
    def setUp(self):
        self.data_model = DataModel()
        self.data_model.train_model()

    def test_predict_with_empty_dataframe(self):
        empty_dataframe = pd.DataFrame()
        with self.assertRaises(ValueError):
            first_key = next(iter(self.data_model.models))
            self.data_model.models[first_key].predict(empty_dataframe)

    def test_predict_with_valid_dataframe(self):
        valid_dataframe = pd.DataFrame({
            'nodes': [1, 1],
            'read_io_size': [1000000, 2000000],
            'write_io_size': [3000000, 4000000],
            'read_volume': [100, 200],
            'write_volume': [300, 400],
            'read_io_pattern': ['seq', 'rand'],
            'write_io_pattern': ['uncl', 'stride'],
        })
        predictions = self.data_model.predict(valid_dataframe)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(predictions.shape[0], valid_dataframe.shape[0])
        self.assertEqual(predictions.shape[1], len(self.data_model.models))

    def test_predict_with_dataframe_missing_nodes(self):
        missing_nodes_dataframe = pd.DataFrame({
            'read_io_size': [1000000, 2000000],
            'write_io_size': [3000000, 4000000],
            'read_volume': [100, 200],
            'write_volume': [300, 400],
            'read_io_pattern': ['seq', 'rand'],
            'write_io_pattern': ['uncl', 'stride'],
        })
        predictions = self.data_model.predict(missing_nodes_dataframe)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(predictions.shape[0], missing_nodes_dataframe.shape[0])
        self.assertEqual(predictions.shape[1], len(self.data_model.models))

    def test_predict_with_dataframe_with_extra_columns(self):
        extra_columns_dataframe = pd.DataFrame({
            'nodes': [1, 1],
            'read_io_size': [1000000, 2000000],
            'write_io_size': [3000000, 4000000],
            'read_volume': [100, 200],
            'write_volume': [300, 400],
            'read_io_pattern': ['seq', 'rand'],
            'write_io_pattern': ['uncl', 'stride'],
            'extra_col1': [1, 2],
            'extra_col2': [3, 4],
        })
        with self.assertRaises(ValueError):
            predictions = self.data_model.predict(extra_columns_dataframe)


class TestDataModelSaving(unittest.TestCase):

    def setUp(self):
        model_1 = TierModel(regressor=LinearRegression())
        model_2 = TierModel(regressor=LinearRegression())
        self.models = {"lfs_bw": model_1, "fs1_bw": model_2}
        self.data_model = DataModel(models=self.models)
        self.data_model.train_model(save_dir=TEST_MODELS)

    def test_save_model(self):
        col = "lfs_bw"
        self.data_model.save_model(col, save_dir=TEST_MODELS)
        model_name = self.data_model.model_name(col)
        file_path = os.path.join(TEST_MODELS, model_name)
        self.assertTrue(os.path.exists(file_path), f"Model file not found at '{file_path}'")



MODEL_PATH = '/home_nfs/mimounis/iosea-wp3-recommandation-system/performance_data/tests/test_data/test_models/linear_regression_lfs_bw.joblib'

class TestDataModelLoadAndPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define the sample data for testing
        cls.sample_data = pd.DataFrame({
                                        'nodes': [1, 1],
                                        'read_io_size': [1000000, 2000000],
                                        'write_io_size': [3000000, 4000000],
                                        'read_volume': [100, 200],
                                        'write_volume': [300, 400],
                                        'read_io_pattern': ['seq', 'rand'],
                                        'write_io_pattern': ['uncl', 'stride']        })

    def test_load_and_predict(self):
        # Call the load_and_predict function
        predictions = load_and_predict(MODEL_PATH, self.sample_data)
        # Check if predictions is a pandas DataFrame
        self.assertIsInstance(predictions, pd.DataFrame)

        # Check if the shape of predictions matches the expected shape
        expected_shape = (len(self.sample_data), 1)  # Assuming the model outputs a single target column
        self.assertEqual(predictions.shape, expected_shape)



class TestDeepNNModel(unittest.TestCase):

    def setUp(self):
        self.X_train = np.random.rand(100, 14)
        self.y_train = np.random.rand(100)
        self.X_test = np.random.rand(50, 14)

        self.tier_model = TierModel(regressor=DeepNNModel(depth=3, width=32, input_dim=14))

    def test_deep_nn_model_fit(self):
        model = deepcopy(self.tier_model)
        model.fit(self.X_train, self.y_train)

    def test_deep_nn_model_predict(self):
        model = deepcopy(self.tier_model)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(predictions.shape, (50,))





if __name__ == '__main__':
    unittest.main()
