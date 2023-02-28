__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import random
import pandas as pd
from performance_data.data_table import PhaseData, DataTable
import pandas as pd
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


class RegressionModel:
    """
    A class for training and evaluating a regression model on performance data.

    Attributes:
        data (pandas.DataFrame): The performance data loaded from the csv file.
        X (pandas.DataFrame): The features of the performance data.
        y (pandas.DataFrame): The targets of the model.
        X_train (pandas.DataFrame): The training features of the performance data.
        X_test (pandas.DataFrame): The test features of the performance data.
        y_train (pandas.DataFrame): The training targets of the model.
        y_test (pandas.DataFrame): The test targets of the model.
        preprocessor (ColumnTransformer): The preprocessor for transforming the data into numerical format.
        model (Pipeline): The pipeline that applies the preprocessor and trains the model.
    """
    def __init__(self, data_file):
        """
        Initializes the RegressionModel with the data file.

        Args:
            data_file (str): The file name of the data in csv format.
        """
        self.data = pd.read_csv(data_file)
        self.X = self.data.filter(regex='_volume$|_io_pattern$|io_size$')
        self.y = self.data.filter(regex='_bw$')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.X.filter(regex='_volume$').columns),
                ("cat", OneHotEncoder(), self.X.filter(regex='_io_pattern$').columns)
            ])
        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", LinearRegression())
        ])

    def train_model(self):
        """
        Trains the regression model on the training data.
        """
        self.model.fit(self.X_train, self.y_train)

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


class RandomForestModel:
    """A class for building a random forest regression model for predicting bandwidth values based on input features.

    Attributes:
        file_path (str): The path to the CSV file containing the input data.
        num_trees (int, optional): The number of trees in the random forest. Defaults to 100.
        max_depth (int, optional): The maximum depth of each tree in the random forest. Defaults to None.

    Methods:
        _create_pipeline(): Create a pipeline object for preprocessing the input data and fitting a random forest regression model.
        fit(): Fit the pipeline to the training data.
        predict(): Make predictions on the test data.
        evaluate(): Evaluate the performance of the model on the test data by calculating mean squared error and R² score.

    """
    def __init__(self, file_path, num_trees=100, max_depth=None):
        """Initialize a RandomForestModel instance.

        Args:
            file_path (str): The path to the CSV file containing the input data.
            num_trees (int, optional): The number of trees in the random forest. Defaults to 100.
            max_depth (int, optional): The maximum depth of each tree in the random forest. Defaults to None.

        """
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.model = RandomForestRegressor(n_estimators=num_trees, max_depth=max_depth, random_state=42)
        self.pipeline = self._create_pipeline()
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.X = self.data.filter(regex='_volume$|_io_pattern$|_io_size$|nodes$')
        self.y = self.data.filter(regex='_bw$')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

    def _create_pipeline(self):
        """Create a pipeline object for preprocessing the input data and fitting a random forest regression model.

        Returns:
            Pipeline: The pipeline object.

        """
        # Define column transformer for preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, ['nodes', 'read_volume', 'write_volume', 'read_io_size', 'write_io_size']),
                ('cat', categorical_transformer, ['read_io_pattern', 'write_io_pattern'])
            ])

        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])
        return pipeline

    def fit(self):
        """Fit the pipeline to the training data."""
        self.pipeline.fit(self.X_train, self.y_train)

    def predict(self):
        """Make predictions on the test data.

        Returns:
            numpy.ndarray: The predicted bandwidth values.

        """
        y_pred = self.pipeline.predict(self.X_test)
        return y_pred

    def evaluate(self):
        """Evaluate the performance of the model on the test data by calculating mean squared error and R² score.

        Returns:
            tuple: A tuple containing the mean squared error and R² score.

        """
        y_pred = self.pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2
