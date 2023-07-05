import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics.pairwise import euclidean_distances
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from numpy import pad


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
        # Convert signal to numpy array
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
        self.features = []
        for job in self.jobs:
            feature_set = []
            for key in ["bytesRead", "bytesWritten"]:
                ts_data = job[key]
                min_val = np.min(ts_data)
                max_val = np.max(ts_data)
                mean_val = np.mean(ts_data)

                # DFT components
                dft_val = self.fft_features(ts_data).ravel().tolist()

                # Append all features to the single list
                feature_set.extend([min_val, max_val, mean_val])
                feature_set.extend(dft_val)

            self.features.append(feature_set)

        return self.features

    def scale_features(self):
        """
        Apply z-score or minmax scaling on the features.
        """
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
        self.process()
        self.scale_features()
        centroid = np.mean(self.features, axis=0)
        distances = euclidean_distances(self.features, [centroid])
        return np.argmin(distances)