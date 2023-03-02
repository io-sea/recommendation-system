__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import random
import re
import os
from abc import ABC, abstractmethod
from performance_data import MODELS_DIRECTORY, DATASET_FILE, GENERATED_DATASET_FILE
from performance_data.data_table import PhaseData, DataTable
import pandas as pd
import joblib
from loguru import logger

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class PhaseGenerator:
    """A class for generating random I/O phases for use in storage system simulations.

    Attributes:
        num_entries (int): The number of data entries to generate.
        volume (float): The total volume of data to generate, in bytes.
        patterns (list): A list of possible I/O patterns.
        io_sizes (list): A list of possible I/O sizes, in bytes.
    """

    def __init__(self, num_entries, volume=10e6):
        """Initialize the data generator.

        Args:
            num_entries (int): The number of data entries to generate.
            volume (float, optional): The total volume of data to generate, in bytes.
                Defaults to 10e6.
        """
        self.num_entries = num_entries
        self.patterns = ["uncl", "seq", "rand", "stride"]
        self.volume = volume
        self.io_sizes = [4e3, 16e3, 128e3, 512e3, 2e6, 8e6]

    def generate_data(self):
        """Generate a list of dictionaries representing storage system data.

        Returns:
            list: A list of dictionaries, each with keys "read_volume", "write_volume",
                "read_io_pattern", "write_io_pattern", "read_io_size", and "write_io_size".
        """
        data = []
        for _ in range(self.num_entries):
            entry = {}
            entry["nodes"] = 1
            entry["read_volume"] = self.volume * random.random()
            entry["write_volume"] = self.volume - entry["read_volume"]
            entry["read_io_pattern"] = random.choice(self.patterns)
            entry["write_io_pattern"] = random.choice(self.patterns)
            entry["read_io_size"] = random.choice(self.io_sizes)
            entry["write_io_size"] = random.choice(self.io_sizes)
            data.append(entry)
        return data

    def export_data(self, filename):
        """Export the generated data as a DataFrame to a CSV file.

        Args:
            filename (str): The name of the file to export the data to.
        """
        df = pd.DataFrame(self.generate_data())
        df.to_csv(filename, index=False)


class AbstractModel(ABC):
    """
    An abstract class for training and evaluating a regression model on performance data.

    Attributes:
        data (pandas.DataFrame): The performance data loaded from the csv file.
        X (pandas.DataFrame): The features of the performance data.
        y (pandas.DataFrame): The targets of the model.
        X_train (pandas.DataFrame): The training features of the performance data.
        X_test (pandas.DataFrame): The test features of the performance data.
        y_train (pandas.DataFrame): The training targets of the model.
        y_test (pandas.DataFrame): The test targets of the model.
        preprocessor (ColumnTransformer): The preprocessor for transforming the data into numerical format.
        model (object): The trained model object.
        model_name (str): The name of the trained model.

    """
    SCORE_THRESHOLD = 0.7

    def __init__(self):
        """
        Initializes the AbstractModel.
        """
        self.data = pd.read_csv(GENERATED_DATASET_FILE)
        assert not self.data.empty, "No elements found in data."
        self.X, self.y = self._prepare_data()
        assert not self.y.empty, "No targets found in data."
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        model_name = re.sub('(?<!^)(?=[A-Z])', '_', f"{type(self).__name__}.joblib").lower()
        # f"{type(self).__name__}.joblib"
        self.model_path = os.path.join(MODELS_DIRECTORY, model_name)
        if os.path.exists(self.model_path):
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
        else:
            logger.info("Creating a new model")
            self.model = self._create_model()

    def _prepare_input_data(self, data):
        """
        Prepares input data for prediction.

        Args:
            data (dict): A dictionary of input data.

        Returns:
            pandas.DataFrame: The prepared input data.
        """
        logger.info("Preparing input data...")
        # extract targets
        target_columns = [col for col in data.columns if col.endswith('_bw')]
        if target_columns:
            data = data.drop(target_columns, axis=1)
        logger.debug(f"Input data after dropping target columns: {data.columns.tolist()}")
        # calculate total volume
        total_volume = data['read_volume'] + data['write_volume']
        # divide read_volume and write_volume by total_volume
        data['read_ratio'] = data['read_volume'] / total_volume
        data['write_ratio'] = data['write_volume'] / total_volume
        # scale read_io_size and write_io_size by 8e6
        data["read_io_size"] = data["read_io_size"] / 8e6
        data["write_io_size"] = data["write_io_size"] / 8e6
        # remove unnecessary columns
        data = data.drop(columns=['read_volume', 'write_volume'], axis=1)
        logger.debug(f"Input data after dropping unnecessary columns: {data.columns.tolist()}")
        # Apply preprocessing to X data
        categorical_cols = data.filter(regex='_io_pattern$').columns
        logger.debug(f"Categorical columns: {categorical_cols.tolist()}")
        preprocessor = ColumnTransformer(
            transformers=[
                #("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(), categorical_cols),
            ],
            remainder="passthrough"
        )

        # transform X data and extract y data
        X = preprocessor.fit_transform(data)
        df = pd.DataFrame(X, columns=list(preprocessor.get_feature_names_out()))
        logger.debug(f"Preprocessed data: {df.head()}")

        return df

    def _prepare_data(self):
        """
        Organizes X and y data by doing some small preprocessing on the loaded dataframe.

        Returns:
            Tuple of (X, y) data.
        """
        logger.info("Preparing data...")
        # extract targets
        target_columns = [col for col in self.data.columns if col.endswith('_bw')]
        y = self.data[target_columns]
        logger.debug(f"Target columns: {target_columns}")
        # extract features
        X = self._prepare_input_data(self.data)
        logger.debug(f"Features: {X.columns.tolist()}")

        return X, y

    @abstractmethod
    def _create_model(self):
        """
        Creates the model object.
        """
        pass

    def train_model(self):
        """
        Trains the regression model on the training data and saves it to disk if the score on the test set is better than a threshold.
        """
        logger.info("Training model...")
        self.model.fit(self.X_train, self.y_train)
        score = self.model.score(self.X_test, self.y_test)
        logger.info(f"Model score: {score}")
        if score > self.SCORE_THRESHOLD:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved with score: {score}")

    def evaluate_model(self):
        """
        Evaluates the model on the test data and returns the score.

        Returns:
            The score of the model on the test data.
        """
        score = self.model.score(self.X_test, self.y_test)
        logger.info(f"Model score on test data: {score}")
        return score


    def predict(self, new_data):
        """
        Makes predictions on new data using the trained model.

        Args:
            new_data (pandas.DataFrame): The new data for which predictions are to be made.

        Returns:
            The predictions made by the model on the new data.
        """
        input_data = self._prepare_input_data(new_data)
        predictions = self.model.predict(input_data)
        logger.info(f"Predictions made by the model: {predictions}")
        return predictions


