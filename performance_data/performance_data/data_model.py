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
    """An abstract class for training and evaluating a regression model on performance data.

    Attributes:
        data: A pandas DataFrame containing the performance data loaded from the CSV file.
        X: A pandas DataFrame containing the features of the performance data.
        y: A pandas DataFrame containing the targets of the model.
        X_train: A pandas DataFrame containing the training features of the performance data.
        X_test: A pandas DataFrame containing the test features of the performance data.
        y_train: A pandas DataFrame containing the training targets of the model.
        y_test: A pandas DataFrame containing the test targets of the model.
        preprocessor: A ColumnTransformer object for transforming the data into numerical format.
        model: A dictionary containing the trained models objects.
        model_name: A string representing the name of the trained model.

    """
    SCORE_THRESHOLD = 0.7

    def __init__(self):
        """Initializes the AbstractModel.

        Raises:
            AssertionError: If no elements found in data.

        """
        # load data
        # TODO: avoid data redundancy, create self.data["X_train"].. self.data["tier"]["y_train"]
        self.input_data = pd.read_csv(GENERATED_DATASET_FILE)
        self.data = {}
        self.model = {}
        # extract list of target tiers
        # target_tiers = [col.split("_bw")[0] for col in data.columns if col.endswith('_bw')]
        self.target_tiers = [col for col in self.input_data.columns if col.endswith('_bw')]
        assert not self.input_data.empty, "No elements found in data."
        for tier_col in self.target_tiers:
            # one model per tier
            self.model[tier_col] = {}
            # one target y per tier
            self.data[tier_col] = {}
            # get dataframes for X and y
            X, y = self._prepare_data(column=tier_col)
            assert not y.empty, "No targets found in data."
            # split data into train and test sets
            self.data["X_train"], self.data["X_test"], self.data[tier_col]["y_train"], self.data[tier_col]["y_test"] = train_test_split(X, y, test_size=0.2, random_state=0)
            # register model name
            self.model[tier_col]["model_name"] = re.sub('(?<!^)(?=[A-Z])', '_', f"{type(self).__name__ + '_' + tier_col}.joblib").lower()
            # register model path
            self.model[tier_col]["model_path"] = os.path.join(MODELS_DIRECTORY, self.model[tier_col]["model_name"])

            # if already registered model, load it
            if os.path.exists(self.model[tier_col]["model_path"]):
                logger.info(f"Loading model from {self.model[tier_col]['model_path']}")
                self.model[tier_col]["model"] = joblib.load(self.model[tier_col]["model_path"])
            else:
                logger.info(f"Creating a new model: {self.model[tier_col]['model_name']}")
                self.model[tier_col]["model"] = self._create_model()

    def _prepare_input_data(self, data):
        """
        Prepares input data for prediction.

        Args:
            data (dict): A dictionary of input data.

        Returns:
            pandas.DataFrame: The prepared input data.
        """
        logger.info("Preparing input data...")
        # dropout targets
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
        data["avg_io_size"] = data["read_io_size"]*data["read_ratio"] + data["write_io_size"]*data["write_ratio"]
        # remove unnecessary columns
        data = data.drop(columns=['read_volume', 'write_volume'], axis=1)
        logger.debug(f"Input data after dropping unnecessary columns: {data.columns.tolist()}")
        # Apply preprocessing to X data
        categorical_cols = data.filter(regex='_io_pattern$').columns
        logger.debug(f"Categorical columns: {categorical_cols.tolist()}")
        preprocessor = ColumnTransformer(
            transformers=[
                # ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(), categorical_cols),
            ],
            remainder="passthrough"
        )

        # transform X data and extract y data
        X = preprocessor.fit_transform(data)
        df = pd.DataFrame(X, columns=list(preprocessor.get_feature_names_out()))
        logger.debug(f"Preprocessed data: {df.head()}")

        return df

    def _prepare_data(self, column=None):
        """
        Organizes X and y data by doing some small preprocessing on the loaded dataframe.

        Args:
            column (str, optional): The target column to extract. Defaults to None.

        Returns:
            Tuple of (X, y) data.
        """
        logger.info("Preparing data...")
        target_columns = column if column and column in self.input_data.columns else [col for col in self.input_data.columns if col.endswith('_bw')]

        logger.debug(f"Target columns: {target_columns}")
        # extract features
        X = self._prepare_input_data(self.input_data)
        y = self.input_data[target_columns].div(X['remainder__avg_io_size'], axis=0)
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
        for target_tier in self.target_tiers:
            logger.info("Training model...")
            self.model[target_tier]["model"].fit(self.data["X_train"], self.data[target_tier]["y_train"])
            self.model[target_tier]["score"] = self.model[target_tier]["model"].score(self.data["X_test"], self.data[target_tier]["y_test"])
            logger.info(f"Model score for tier {target_tier}: {self.model[target_tier]['score']}")

            if self.model[target_tier]["score"] > self.SCORE_THRESHOLD:
                if not os.path.exists(os.path.dirname(self.model[target_tier]["model_path"])):
                    os.makedirs(os.path.dirname(self.model[target_tier]["model_path"]))
                joblib.dump(self.model[target_tier]["model"], self.model[target_tier]["model_path"])
                logger.info(f"Saving Model for {target_tier}: {self.model[target_tier]['model_path']} | saved with score: {self.model[target_tier]['score']}")

    def evaluate_model(self):
        """
        Evaluates the model on the test data and returns the score.

        Returns:
            The score of the model on the test data.
        """
        for target_tier in self.target_tiers:
            self.model[target_tier]["score"] = self.model[target_tier]["model"].score(self.data["X_test"], self.data[target_tier]["y_test"])
            logger.info(f"Model: {self.model[target_tier]['model_name']} | Score on test data: {self.model[target_tier]['score']}")

    def predict(self, new_data, process_input=True):
        """
        Makes predictions on new data using the trained model.

        Args:
            new_data (pandas.DataFrame): The new data for which predictions are to be made.

        Returns:
            The predictions made by the model on the new data.
        """
        predictions = {}
        input_data = self._prepare_input_data(new_data) if process_input else new_data
        for target_tier in self.target_tiers:
            predictions[target_tier] = self.model[target_tier]["model"].predict(input_data)
            logger.trace(f"Predictions made by the model {self.model[target_tier]['model_name']}: {predictions}")
        return predictions
