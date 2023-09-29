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
import numpy as np
import joblib
from loguru import logger

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import RegressorMixin


DEFAULT_CATEGORIES = {"rand", "stride", "seq", "uncl"}

def load_model(model_path):
    """
    Loads a trained model from file.

    Args:
        model_path (str): The path to the saved model.

    Returns:
        The loaded model object.
    """
    return joblib.load(model_path)


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


class RegressionModel(ABC, RegressorMixin):
    """Abstract base class for regression models."""

    @abstractmethod
    def fit(self, X, y):
        """Fit the regression model to the training data.

        Args:
            X: array-like or sparse matrix of shape (n_samples, n_features).
                Training data.
            y: array-like of shape (n_samples,).
                Target values.

        Returns:
            self: object.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict target values for the given test data.

        Args:
            X: array-like or sparse matrix of shape (n_samples, n_features).
                Test data.

        Returns:
            y_pred: array-like of shape (n_samples,).
                Predicted target values.
        """
        pass


class TierModel(RegressionModel):
    """
    Concrete regression model that implements the abstract base class RegressionModel using scikit-learn's LinearRegression class or any other regressor that can be passed as an argument.

    Attributes:
        model: object, optional (default=LinearRegression())
            Regression model object to be used for fitting the data.

    Methods:
        fit(X, y):
            Fits the regression model to the training data.
        predict(X):
            Predicts target values for the given test data.
    """

    def __init__(self, regressor=LinearRegression()):
        """
        Initializes the TierModel object with a specified regression model object.

        Parameters:
            regressor: object, optional (default=LinearRegression())
                Regression model object to be used for fitting the data.
        """
        self.model = regressor

    def __call__(self, X):
        return self.model.predict(X)

    def fit(self, X, y):
        """
        Fits the regression model to the training data.

        Parameters:
            X: array-like or sparse matrix of shape (n_samples, n_features)
                Training data.
            y: array-like of shape (n_samples,)
                Target values.

        Returns:
            self: object
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts target values for the given test data.

        Parameters:
            X: array-like or sparse matrix of shape (n_samples, n_features)
                Test data.

        Returns:
            y_pred: array-like of shape (n_samples,)
                Predicted target values.
        """
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
            deep: bool, default=True
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            params: dict
                Parameter names mapped to their values.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters:
            **params: dict
                Estimator parameters.

        Returns:
            self: object
                Estimator instance.
        """
        self.model.set_params(**params)
        return self



class DataModel:
    """Handles data loading, preprocessing, model training, and prediction.

    Class is designed to work with data from a CSV file and can train
    multiple `TierModel` instances.

    Attributes:
        X: pandas.DataFrame
            Input data with shape (n_samples, n_features).
        y: pandas.DataFrame
            Target data with shape (n_samples, n_targets).

    Args:
        data_file: str
            Path to the CSV data file. Defaults to `GENERATED_DATASET_FILE`.
        cats: list
            List of categories. Defaults to `DEFAULT_CATEGORIES`.
        models: dict or list or None
            Predefined models to train. If a dictionary is passed, keys must
            match target columns. If a list is passed, it must match the number
            of target columns. If None, default models are initialized.
            Defaults to None.

    Raises:
        TypeError: If `models` is not a dictionary, a list, or None.
    """
    THRESHOLD = 0.7

    def __init__(self,
                 data_file=GENERATED_DATASET_FILE,
                 cats=DEFAULT_CATEGORIES,
                 models=None):
        logger.info("Initializing DataModel object.")
        self.data_file = data_file
        self.cats = cats
        self.X = None
        self.y = None

        # load data
        self.load_data()
        self.X, self.y = self._prepare_data()

        self.models = {}
        logger.info("Initializing models.")

        if models is None:
            logger.debug("No models provided. Initializing default models.")
            self.models = {col: TierModel(regressor=LinearRegression()) for col in self.y.columns}

        elif isinstance(models, dict):
            logger.debug("Models provided as a dictionary. Initializing models.")
            for key, model in models.items():
                if key in self.y.columns:
                    self.models[key] = model
                    logger.debug(f"Model for '{key}' initialized.")
                else:
                    self.models[key] = TierModel(regressor=LinearRegression())
                    logger.debug(f"Model for '{key}' not provided, initialized as default.")

        elif isinstance(models, list):
            logger.debug("Models provided as a list. Initializing models.")
            for i, model in enumerate(models):
                self.models[self.y.columns[i]] = model
                logger.debug(f"Model for '{self.y.columns[i]}' initialized.")
        else:
            logger.error("Invalid value for 'models' parameter. Must be a dictionary, a list, or None.")
            raise TypeError("Invalid value for 'models' parameter. Must be a dictionary, a list, or None.")

    def load_data(self):
        """
        Loads data from a CSV file.

        Parameters:
            file_path: str
                Path to the CSV file containing input and target data.

        Returns:
            None
        """
        logger.info(f"Ingesting input data.")
        self.input_data = pd.read_csv(self.data_file)
        self.target_tiers = [col for col in self.input_data.columns if col.endswith('_bw')]
        logger.info(f"Target tiers: {self.target_tiers}")
        assert not self.input_data.empty, "No elements found in data."

    @staticmethod
    def _prepare_input_data(data, all_categories=["uncl", "seq", "rand", "stride"]):
        """
        The _prepare_input_data method prepares input data for prediction by performing several data preprocessing steps. It takes in a dictionary of input data and returns a pandas DataFrame of the prepared input data.

        The method first drops the target columns from the input data, if they exist. It then calculates the total volume of read and write operations and divides the read and write volumes by the total volume. The read and write I/O sizes are then scaled by 8e6 and the average I/O size is calculated using the scaled read and write I/O sizes and the read and write ratios. The unnecessary columns are then dropped and the categorical columns are identified for preprocessing. A ColumnTransformer object is then used to apply preprocessing to the X data, including standard scaling of numeric columns and one-hot encoding of categorical columns. The transformed X data is returned as a pandas DataFrame of the prepared input data.

        Args:
            data (dict): A dictionary of input data.

        Returns:
            pandas.DataFrame: The prepared input data.
        """
        data = data.copy()
        logger.trace(f"Preparing input data with columns: {list(data.columns)}")
        # dropout targets
        target_columns = [col for col in data.columns if col.endswith('_bw')]
        if target_columns:
            data = data.drop(target_columns, axis=1)
        logger.trace(f"Input data after dropping target columns: {data.columns.tolist()}")
        # calculate total volume
        total_volume = data['read_volume'] + data['write_volume']
        # divide read_volume and write_volume by total_volume
        data['read_ratio'] = (data['read_volume'] / total_volume).fillna(0)
        data['write_ratio'] = (data['write_volume'] / total_volume).fillna(0)
        # scale read_io_size and write_io_size by 8e6
        data["read_io_size"] = data["read_io_size"] / 8e6
        data["write_io_size"] = data["write_io_size"] / 8e6
        data["avg_io_size"] = (data["read_io_size"]*data["read_ratio"] + data["write_io_size"]*data["write_ratio"]).fillna(0)
        # remove unnecessary columns
        data = data.drop(columns=['read_volume', 'write_volume'], axis=1)
        logger.trace(f"Input data after dropping unnecessary columns: {data.columns.tolist()}")
        # Apply preprocessing to X data
        categorical_cols = data.filter(regex='_io_pattern$').columns
        logger.trace(f"Categorical columns: {categorical_cols.tolist()}")

        # Update category dictionary with new categories
        category_dict = {}
        for col in data.columns:
            if col.endswith("_io_pattern"):
                categories = all_categories #self.cats
                if col in category_dict:
                    category_dict[col].update(categories)
                else:
                    category_dict[col] = categories

        # Get all categories
        all_categories = [list(category_dict.get(col, set())) for col in data.columns if col.endswith("_io_pattern")]

        preprocessor = ColumnTransformer(
            transformers=[
                # ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(categories=all_categories), categorical_cols),
            ],
            remainder="passthrough"
        )

        # transform X data and extract y data
        X = preprocessor.fit_transform(data)
        df = pd.DataFrame(X, columns=list(preprocessor.get_feature_names_out()))
        logger.trace(f"Preprocessed input data: {df.columns.tolist()}")

        return df

    def _prepare_data(self, column=None):
        """
        Organizes X and y data by doing some small preprocessing on the loaded dataframe.
        The _prepare_data method of a class prepares the input data for training by organizing X and y data with some small preprocessing.

        The method takes an optional argument column which specifies the target column to extract. If column is not provided or is not found in the loaded dataframe, the method looks for target columns with names ending in _bw.

        The method then extracts the features from the input data using the _prepare_input_data method and computes the target values y by dividing the target column(s) by the average IO size of the input data.

        The method returns a tuple of X and y data.

        Args:
            column (str, optional): The target column to extract. Defaults to None.

        Returns:
            Tuple of (X, y) data.
                X: Pandas DataFrame of input data features.
                        Example:

                            col_1   col_2   col_3
                            0    0.1     0.5     0.8
                            1    0.2     0.4     0.7
                            2    0.3     0.5     0.6

                    y: Pandas DataFrame of target values.
                        Example:

                            target_1    target_2
                            0    2.3         4.5
                            1    3.1         5.6
                            2    1.8         3.2
        """
        logger.info("Preparing data...")
        target_columns = column if column and column in self.input_data.columns else [col for col in self.input_data.columns if col.endswith('_bw')]

        zero_factor = self.input_data.apply(lambda row: 0 if row["read_volume"] == 0 and row["write_volume"] == 0 else 1, axis=1)
        logger.trace(f"Target columns: {target_columns}")
        # extract features
        X = DataModel._prepare_input_data(self.input_data)
        y = self.input_data[target_columns].multiply(zero_factor, axis=0).div(X['remainder__avg_io_size'].replace(0, np.nan), axis=0).fillna(0)

        # Find the maximum value for each column in the target_columns
        max_values = y[target_columns].max()

        # Normalize y by dividing each column with its respective maximum value
        #y_normalized = y.div(max_values, axis=1)
        logger.trace(f"Features: {X.columns.tolist()}")
        #logger.info(f"Max values: {max_values}")

        return X, y

    def model_name(self, col):
        # model corresponds to self.models[col], col
        # name is the name of the model class with lower case and underscores
        return re.sub('(?<!^)(?=[A-Z])', '_',
                      f"{type(self.models[col].model).__name__ + '_' + col}.joblib").lower()

    def train_model(self, test_size=0.2, random_state=None, save_dir=MODELS_DIRECTORY):
        """
        Trains TierModel instances for each tier column in the target data using the training dataset
        and evaluates their performance using the testing dataset.

        Parameters:
            test_size: float, default=0.2
                The proportion of the dataset to include in the test split.
            random_state: int or None, default=None
                Controls the shuffling applied to the data before applying the split.
                Pass an int for reproducible output across multiple function calls.

        Returns:
            models: dictionary
                Dictionary containing trained TierModel instances, one per target column in the target data.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
        logger.info(f"Models {self.y.columns}")
        for col in self.y.columns:

            # self.models[col] = self.models[col].fit(self.X_train, self.y_train[col])
            # # Compute scores for each model
            # score = self.models[col].score(self.X_test, self.y_test[col])
            scores = []

            for train_index, test_index in kfold.split(self.X_train):
                X_train, X_test = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
                y_train, y_test = self.y_train[col].iloc[train_index], self.y_train[col].iloc[test_index]
                self.models[col] = self.models[col].fit(X_train, y_train)
                score = self.models[col].score(X_test, y_test)
                scores.append(score)

            score = np.mean(scores)
            logger.info(f"Model {self.models[col].model} for Tier: {col} trained with score:{score:.3}")
            # Save the model if its score is above the threshold
            if score > self.THRESHOLD:
                self.save_model(col, save_dir)
        return self.models

    def save_model(self, col, save_dir=None):
        """
        Saves a trained model to the specified directory using joblib.

        Parameters:
            col: str
                The target column name, which is used to identify the corresponding trained model.

        Returns:
            None
        """
        model_name = self.model_name(col)
        model = self.models[col]
        save_dir = save_dir if save_dir else self.MODELS_DIRECTORY

        # Create the save_dir directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        file_path = os.path.join(save_dir, model_name)
        joblib.dump(model, file_path)
        logger.info(f"Model {model_name} for '{col}' saved at '{file_path}'")

    def predict(self, input_data):
        """
        Predicts the target values for input data.

        Parameters:
            input_data: pandas DataFrame of shape (n_samples, n_features)
                Input data.

        Returns:
            pandas DataFrame of shape (n_samples, n_targets)
                Predicted target values.
        """
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        # Add 'nodes' column if it doesn't exist
        if "nodes" not in input_data.columns:
            input_data["nodes"] = 1
        # Prepare input data
        X = DataModel._prepare_input_data(input_data)
        # Predict target values for each model
        predictions = {}
        for col, model in self.models.items():
            y_pred = model.predict(X)
            predictions[col] = y_pred

        return pd.DataFrame(predictions)


# class DeepNNModel(TierModel):
#     """
#     Subclass of TierModel that creates a deep neural network (NN) model for regression tasks.

#     Attributes:
#         depth: int
#             The number of hidden layers in the neural network.
#         width: int
#             The number of neurons in each hidden layer.
#         input_dim: int
#             The number of input features for the neural network.
#     """
#     def __init__(self, depth=15, width=32, input_dim=14, epochs=50, **kwargs):
#         """
#         Initializes the DeepNNModel object with the specified depth and width.

#         Parameters:
#             depth: int
#                 The number of hidden layers in the neural network.
#             width: int
#                 The number of neurons in each hidden layer.
#             input_dim: int
#                 The number of input features for the neural network.
#             **kwargs: dict
#                 Additional keyword arguments for KerasRegressor.
#         """
#         self.depth = depth
#         self.width = width
#         self.input_dim = input_dim
#         self.epochs = epochs
#         super().__init__(regressor=self._build_nn_model(), **kwargs)

#     def _build_nn_model(self):
#         """
#         Builds the neural network model using Keras.

#         Returns:
#             nn_model: KerasRegressor
#                 The KerasRegressor object wrapping the neural network model.
#         """
#         def create_model():
#             model = Sequential()
#             model.add(Dense(self.width, input_dim=self.input_dim, activation='relu'))

#             for _ in range(self.depth - 1):
#                 model.add(Dense(self.width, activation='relu'))

#             model.add(Dense(1, activation='linear'))
#             model.compile(loss='mean_squared_error', optimizer='adam')
#             return model

#         nn_model = KerasRegressor(build_fn=create_model, epochs=self.epochs)
#         return nn_model

def load_and_predict(model_path, new_data, iops=False):
    """
    Load a model from a joblib file and use it to predict on new data.

    Parameters:
        model_path: str
            The path to the joblib file containing the trained model.
        new_data: pandas DataFrame
            The new data for which to make predictions.
        iops: bool, optional, default=False
            If True, adjust the predictions based on the 'remainder__avg_io_size' column
            in the input data. This is useful when predicting IOPS instead of bandwidth.

    Returns:
        pandas DataFrame: The predicted values.
    """
    # Load the model from the joblib file
    model = joblib.load(model_path)

    # Sanitize new data
    new_data = new_data[["nodes", "read_volume", "write_volume", "read_io_pattern",
                         "write_io_pattern", "read_io_size", "write_io_size"]]

    # Check if new_data is a pandas DataFrame
    if not isinstance(new_data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # Prepare input data
    X = DataModel._prepare_input_data(new_data)

    # Make predictions
    model_predictions = pd.DataFrame(model.predict(X))
    #print(f"Model: {model_path} | Input: {new_data} | Predictions: {model_predictions}")
    # Adjust predictions if iops is True
    if iops:
        predictions = model_predictions * X['remainder__avg_io_size'].values.reshape(-1, 1)
    else:
        predictions = model_predictions

    # Set predictions to 0 if both read_volume and write_volume are 0
    predictions.loc[(new_data['read_volume'] == 0) & (new_data['write_volume'] == 0)] = 0

    return predictions

