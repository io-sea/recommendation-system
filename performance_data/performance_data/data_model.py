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

        self.model_path = os.path.join(MODELS_DIRECTORY, f"{type(self).__name__}.joblib")
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = self._create_model()

    def _prepare_data(self):
        """
        Organizes X and y data by doing some small preprocessing on the loaded dataframe.

        Returns:
            Tuple of (X, y) data.
        """
        # extract targets
        target_columns = [col for col in self.data.columns if col.endswith('_bw')]
        y = self.data[target_columns]
        # calculate total volume
        total_volume = self.data['read_volume'] + self.data['write_volume']
        # divide read_volume and write_volume by total_volume
        self.data['read_ratio'] = self.data['read_volume'] / total_volume
        self.data['write_ratio'] = self.data['write_volume'] / total_volume
        # scale read_io_size and write_io_size by 8e6
        self.data["read_io_size"] = self.data["read_io_size"] / 8e6
        self.data["write_io_size"] = self.data["write_io_size"] / 8e6
        # remove unnecessary columns
        self.data = self.data.drop(columns=['read_volume', 'write_volume'] + target_columns, axis=1)

        # separate columns to apply different transformations
        #numeric_cols = ["total_volume", "read_io_size", "write_io_size"]
        categorical_cols = self.data.filter(regex='_io_pattern$').columns

        # Apply preprocessing to X data
        preprocessor = ColumnTransformer(
            transformers=[
                #("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(), categorical_cols),
            ],
            remainder="passthrough"
        )

        # transform X data and extract y data
        X = preprocessor.fit_transform(self.data)
        X = pd.DataFrame(X, columns=list(preprocessor.get_feature_names_out()))
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
        self.model.fit(self.X_train, self.y_train)
        score = self.model.score(self.X_test, self.y_test)
        if score > self.SCORE_THRESHOLD:
            joblib.dump(self.model, self.model_path)
            print(f"Model saved with score: {score}")

    def evaluate_model(self):
        """
        Evaluates the model on the test data and returns the score.

        Returns:
            The score of the model on the test data.
        """
        score = self.model.score(self.X_test, self.y_test)
        return score

    def predict(self, new_data):
        """
        Makes predictions on new data using the trained model.

        Args:
            new_data (pandas.DataFrame): The new data for which predictions are to be made.

        Returns:
            The predictions made by the model on the new data.
        """
        predictions = self.model.predict(new_data)
        return predictions


