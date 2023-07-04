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

    def __init__(self, jobs, n_dft_coeff=20, normalization_type='zscore'):
        """
        Initialize CentralJob instance.

        Args:
            jobs (list): List of jobs data.
            n_dft_coeff (int): Number of DFT coefficients to extract from the job.
            normalization_type (str): The type of normalization to apply to the features.
        """
        self.jobs = jobs
        self.n_dft_coeff = n_dft_coeff
        self.normalization_type = normalization_type
        self.features = None

    def extract_features(self):
        """
        Extracts features (min, max, mean, DFT components) from jobs data.
        """
        self.features = []
        for job in self.jobs:
            min_br = np.min(job['bytesRead'])
            max_br = np.max(job['bytesRead'])
            mean_br = np.mean(job['bytesRead'])
            
            min_bw = np.min(job['bytesWritten'])
            max_bw = np.max(job['bytesWritten'])
            mean_bw = np.mean(job['bytesWritten'])

            # DFT components
            dft_br = fft(job['bytesRead'])[:self.n_dft_coeff]
            dft_bw = fft(job['bytesWritten'])[:self.n_dft_coeff]

            self.features.append([min_br, max_br, mean_br, min_bw, max_bw, 
                                  mean_bw, *dft_br, *dft_bw])

    def scale_features(self):
        """
        Apply z-score or minmax scaling on the features.
        """
        if self.normalization_type == 'zscore':
            self.features = zscore(self.features, axis=0)
        elif self.normalization_type == 'minmax':
            scaler = MinMaxScaler()
            self.features = scaler.fit_transform(self.features)

    def find_central_job(self):
        """
        Finds the job closest to the centroid based on Euclidean distance. 
        Returns the index of the closest job in the jobs list.
        """
        self.extract_features()
        self.scale_features()
        centroid = np.mean(self.features, axis=0)
        distances = euclidean_distances(self.features, [centroid])
        return np.argmin(distances)
