import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import zscore
from sklearn.metrics.pairwise import euclidean_distances
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
import plotly.graph_objs as go
from sklearn.decomposition import PCA


class WorkflowSynthesizer:
    """
    Class for synthesizing a workflow from a list of jobs. The jobs and the 
    workflow are represented as dataframes with time-series data.
    """

    def __init__(self):
        """
        Initialize a new instance of the WorkflowSynthesizer class.
        """
        self.workflow = pd.DataFrame()

    def synthetize(self, jobs):
        """
        Synthesize a workflow from the provided jobs.

        Args:
            jobs (list): A list of jobs where each job is a dictionary with 
                keys 'timestamp', 'bytesRead', and 'bytesWritten'.
        """
        dfs = []
        for i, job in enumerate(jobs):
            df = pd.DataFrame({
                f'timestamp_{i+1}': job['timestamp'],
                f'bytesRead_{i+1}': job['bytesRead'],
                f'bytesWritten_{i+1}': job['bytesWritten']
            })
            df = df.set_index(f'timestamp_{i+1}')
            dfs.append(df)
        self.workflow = pd.concat(dfs, axis=1).fillna(0)
        self.workflow['sumBytesRead'] = self.workflow.filter(
            regex=("bytesRead_")).sum(axis=1)
        self.workflow['sumBytesWritten'] = self.workflow.filter(
            regex=("bytesWritten_")).sum(axis=1)
        self.workflow.reset_index(inplace=True)

    def to_dict(self):
        """
        Convert the synthesized workflow to a dictionary format.

        Returns:
            dict: The synthesized workflow in dictionary format.
        """
        output = {}
        output['timestamp'] = self.workflow['index'].tolist()
        output['bytesRead'] = self.workflow['sumBytesRead'].tolist()
        output['bytesWritten'] = self.workflow['sumBytesWritten'].tolist()
        return output


class WorkflowSearcher:
    def __init__(self, connector):
        self.connector = connector

    def search_workflows(self, workflow_name):
        # Set the endpoint for the request
        endpoint = "/ioi/workflows/"

        # Set the parameters for the request
        params = {
            "filtering": [
                {
                    "field": "name",
                    "comparator": "equals",
                    "comparison_value": workflow_name
                }
            ],
            "order": "asc",
            "sorting_field": "startTime",
            "limit": 50,
            "offset": 0
        }

        # Use the request_delegator method to make the POST request
        response = self.connector.request_delegator("POST", endpoint,
                                                    input_json=params)

        # Extract the list of workflow IDs
        workflow_data = response.json()['data']
        df = pd.DataFrame(workflow_data)
        return df

    def extract_workflow_data(self, workflow_id):
        # Set the endpoint and parameters for the request
        endpoint = f"/ioi/series/workflow/{workflow_id}"
        params = {"metrics_group": "volume"}

        # Use the request_delegator method to make the GET request
        response = self.connector.request_delegator("GET", endpoint, params=params)

        data = response.json()
        converted_data = {
            workflow_id: {
                "bytesRead": np.array([item["bytesRead"] for item in data]),
                "bytesWritten": np.array([item["bytesWritten"] for item in data]),
                "timestamp": np.array([item["timestamp"] for item in data]),
            }
        }
        
        return converted_data


class CentralJob:
    """
    This class is used to identify the job closest to the centroid from a set of jobs. 
    The job is found by extracting certain features from the jobs, normalizing these features,
    determining the centroid, and identifying the job closest to the centroid based on Euclidean distance. 

    Attributes:
        n_dft_coeff: The number of DFT coefficients to extract from the job.
        normalization_type: The type of normalization to apply to the features. Options are 'zscore' and 'minmax'.
        jobs: The jobs to analyze.
        features: The features extracted from the jobs. 
    """

    def __init__(self, jobs, n_components=20, normalization_type='minmax'):
        """
        Initialize CentralJob instance.

        Args:
            jobs (list): List of jobs data.
            n_components (int): Number of DFT coefficients to extract from the job.
            normalization_type (str): The type of normalization to apply to the features.
        """
        logger.info("Initializing CentralJob instance")
        self.jobs = jobs
        self.n_components = n_components
        self.normalization_type = normalization_type
        self.features = None

    def fft_features(self, signal, fixed_length=None):
        """
        Combines real and complex coefficients to give the norm, also called energy
        of a signal, by using the fast numpy.fft implementation.
        It operates on row-wise stacked vectors.

        Args:
            signal (list): row-wise signal stacked vertically.
            fixed_length (int): final length (axis=1 size) of the signal before fft is applied. It pads
                signal from its original length to a fixed_length with zeros. Default, None.

        Returns:
            numpy array: signal energy on N points.
        """
        logger.info("Calculating FFT features")
        signal = np.array(signal)[np.newaxis, :]
        signal_length = signal.shape[1]
        if fixed_length:
            fixed_length = max(fixed_length - signal_length, 0)
            signal = np.pad(signal, ((0, 0), (0, fixed_length)), mode='constant', constant_values=0)
        fft_data = np.fft.fft(signal, n=self.n_components, axis=1)
        fft_data = (2/signal_length) * np.abs(fft_data)[:, 0:self.n_components]
        return fft_data

    def process(self):
        """
        Perform frequency based feature extraction for each job.
        Extracts features (min, max, mean, FFT components) from jobs data.

        Returns:
            list: list of extracted features.
        """
        logger.info("Processing jobs data")
        self.features = []
        for job_id, job in self.jobs.items():
            feature_set = []
            for key in ["bytesRead", "bytesWritten"]:
                ts_data = job[key]
                min_val = np.min(ts_data)
                max_val = np.max(ts_data)
                mean_val = np.mean(ts_data)

                dft_val = self.fft_features(ts_data).ravel().tolist()
                feature_set.extend([min_val, max_val, mean_val])
                feature_set.extend(dft_val)

            self.features.append(feature_set)

        return self.features

    def scale_features(self):
        """
        Apply z-score or minmax scaling on the features.
        """
        logger.info("Scaling features")
        features_df = pd.DataFrame(self.features)
        if self.normalization_type == 'zscore':
            self.features = zscore(features_df, axis=0)
        elif self.normalization_type == 'minmax':
            scaler = MinMaxScaler()
            self.features = scaler.fit_transform(features_df)

    def find_central_job(self):
        """
        Finds the job closest to the centroid based on Euclidean distance. 
        Returns the index of the closest job in the jobs list.
        """
        logger.info("Finding central job")
        self.process()
        self.scale_features()
        centroid = np.mean(self.features, axis=0)
        distances = euclidean_distances(self.features, [centroid])
        central_job_idx = np.argmin(distances)
        logger.info(f"Central job found at index {central_job_idx}")
        return central_job_idx
    

def display_features_3d(features):
    """
    Display the features in 3D using PCA and Plotly. The closest job to the centroid is highlighted.
    """
    # Perform PCA on the features
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(features)

    # Calculate the variance explained by each principal component
    explained_variance = pca.explained_variance_ratio_

    # Calculate the centroid
    centroid = np.mean(reduced_features, axis=0)

    # Find the index of the central job
    distances = euclidean_distances(reduced_features, [centroid])
    central_idx = np.argmin(distances)

    # Create a Plotly figure
    fig = go.Figure()

    # Add each job's features to the figure as a scatter plot
    for i, feature in enumerate(reduced_features):
        color = 'blue'
        size = 6
        if i == central_idx:
            # Highlight the central job
            color = 'red'
            size = 8
        fig.add_trace(go.Scatter3d(
            x=[feature[0]],
            y=[feature[1]],
            z=[feature[2]],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
            ),
            text=[f'Job {i}'],
            name=f'Job {i}'
        ))

    # Add the centroid to the figure
    fig.add_trace(go.Scatter3d(
        x=[centroid[0]],
        y=[centroid[1]],
        z=[centroid[2]],
        mode='markers',
        marker=dict(
            size=8,
            color='black',
        ),
        name='Centroid'
    ))

    # Add the variance explained by each principal component to the figure's title
    fig.update_layout(
        title=f"3D plot of features (PCA~: {np.sum(explained_variance)})"
    )

    # Return the figure
    return fig