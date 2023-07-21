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
import plotly.subplots as sp
import math
from sklearn.manifold import TSNE




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
                f'bytesRead_{i+1}': np.array(job['bytesRead'], dtype=np.int64),
                f'bytesWritten_{i+1}': np.array(job['bytesWritten'], dtype=np.int64)
            })
            df = df.set_index(f'timestamp_{i+1}')
            dfs.append(df)
        self.workflow = pd.concat(dfs, axis=1).fillna(0)
        self.workflow['sumBytesRead'] = self.workflow.filter(regex=("bytesRead_")).sum(axis=1)
        self.workflow['sumBytesWritten'] = self.workflow.filter(regex=("bytesWritten_")).sum(axis=1)
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

        try:
            # Use the request_delegator method to make the GET request
            response = self.connector.request_delegator("GET", endpoint, params=params)
            
            # Check for 500 error
            response.raise_for_status()  # This will raise an HTTPError if the response was an error

            data = response.json()
            converted_data = {
                workflow_id: {
                    "bytesRead": np.array([item["bytesRead"] for item in data]),
                    "bytesWritten": np.array([item["bytesWritten"] for item in data]),
                    "timestamp": np.array([item["timestamp"] for item in data]),
                }
            }

            return converted_data        

        except Exception as err:
            print(f"An error occurred: {err}")
            return None


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
            jobs (dict): Dictionary of jobs data where keys are job IDs and values are job data.
            n_components (int): Number of DFT coefficients to extract from the job.
            normalization_type (str): The type of normalization to apply to the features.
        """
        logger.info("Initializing CentralJob instance")
        self.jobs = jobs
        self.n_components = n_components
        self.normalization_type = normalization_type
        self._features = None

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
    
    @property
    def features(self):
        """
        Returns the features DataFrame, excluding the job_id column used for plotting.
        """
        return self._features.drop(columns=["job_id"])

    def process(self):
        """
        Perform frequency based feature extraction for each job.
        Extracts features (min, max, mean, FFT components) from jobs data.

        Returns:
            DataFrame: DataFrame of extracted features, indexed by job_id.
        """
        logger.info("Processing jobs data")
        features = []
        for job_id, job in self.jobs.items():
            feature_set = {'job_id': job_id}
            for key in ["bytesRead", "bytesWritten"]:
                ts_data = job[key]
                min_val = np.min(ts_data)
                max_val = np.max(ts_data)
                mean_val = np.mean(ts_data)

                dft_val = self.fft_features(ts_data).ravel().tolist()
                feature_set.update({f'{key}_{stat}': stat_val for stat, stat_val in zip(['min', 'max', 'mean'], [min_val, max_val, mean_val])})
                feature_set.update({f'{key}_dft_{i}': val for i, val in enumerate(dft_val)})

            features.append(feature_set)

        self._features = pd.DataFrame(features).set_index("job_id")

    def scale_features(self):
        """
        Apply z-score or minmax scaling on the features.
        """
        logger.info("Scaling features")
        if self.normalization_type == 'zscore':
            self._features = self._features.apply(zscore)
        elif self.normalization_type == 'minmax':
            scaler = MinMaxScaler()
            self._features = pd.DataFrame(scaler.fit_transform(self._features), columns=self._features.columns, index=self._features.index)

    def find_central_job(self):
        """        
        Finds the job closest to the centroid based on Euclidean distance. 
        Returns the job_id of the closest job.
        """
        logger.info("Finding central job")
        self.process()
        self.scale_features()
        centroid = self._features.mean(axis=0)
        distances = euclidean_distances(self._features, [centroid])
        central_job_id = self._features.index[np.argmin(distances)]
        logger.info(f"Central job found with ID {central_job_id}")
        return central_job_id
  



def display_features(features, central_job_id, dim_reduction='PCA', seed=42, 
                     n_components=3, add_centroid=True):
    """
    Display the features using PCA, t-SNE and Plotly. The job closest to the centroid is highlighted.
    """
    
    def add_centroid_to_features(df, job_id="centroid"):
        """Adds a virtual job to the dataframe with features being the average of all other jobs."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        centroid_features = df[numeric_cols].mean(axis=0)
        centroid_df = pd.DataFrame([centroid_features], columns=df.columns)
        centroid_df['job_id'] = job_id
        return pd.concat([df, centroid_df])


    # Reset index
    features_df = features.reset_index()
    if add_centroid:
        features_df = add_centroid_to_features(features_df)
    # Separate features from job_id
    job_ids = features_df['job_id'].values
    features = features_df.drop(columns='job_id')

    # Perform dimensionality reduction
    if dim_reduction == 'PCA':
        reducer = PCA(n_components=n_components, random_state=seed)
    elif dim_reduction == 't-SNE':
        if features.shape[0] <= n_components:  # set perplexity value to be less than the number of samples
            perplexity = features.shape[0] - 1
        else:
            perplexity = n_components
        reducer = TSNE(n_components=n_components, random_state=seed, perplexity=perplexity)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {dim_reduction}")

    reduced_features = reducer.fit_transform(features)

    if add_centroid:
        # Create a dataframe from the reduced features
        reduced_features_df = pd.DataFrame(reduced_features, 
                                        index=job_ids, 
                                        columns=[f"component_{i}" for i in range(1, n_components + 1)])
        # Add a reduced_centroid to the dataframe
        numeric_cols = reduced_features_df.columns
        reduced_centroid_features = reduced_features_df.mean(axis=0)
        reduced_centroid_df = pd.DataFrame(reduced_centroid_features).T  # Transpose to get it as a single row
        reduced_centroid_df.index = ["reduced_centroid"]  # Set index value to 'reduced_centroid'
        
        # Append the reduced_centroid_df to the main dataframe
        reduced_features_df = pd.concat([reduced_features_df, reduced_centroid_df])
        # Update the reduced_features and job_ids variables
        reduced_features = reduced_features_df.values
        job_ids = reduced_features_df.index.values


    # if add_centroid:
    #     reduced_features_df = pd.DataFrame(reduced_features, 
    #                                        index=job_ids, 
    #                                        columns=[f"component_{i}" for i in range(1, n_components + 1)])
    #     print(reduced_features_df)
    #     reduced_features_df = add_centroid_to_features(reduced_features_df,
    #                                                    job_id="reduced_centroid")
    #     print(reduced_features_df)
    #     reduced_features = reduced_features_df.to_numpy()
    #     job_ids = reduced_features_df.index.values

    # Calculate the variance explained by each principal component (if applicable)
    explained_variance = np.sum(reducer.explained_variance_ratio_) if dim_reduction == 'PCA' else None

    # Create a Plotly figure
    fig = go.Figure()

    # Add each job's features to the figure as a scatter plot
    for i, (job_id, feature) in enumerate(zip(job_ids, reduced_features)):
        color = 'blue'
        size = 6
        if job_id == central_job_id:  # Highlight the central job
            color = 'red'
            size = 8
        elif job_id in ["centroid", "reduced_centroid"]:  # Highlight the centroid
            color = 'gray'
            size = 8
        if n_components == 3:
            fig.add_trace(go.Scatter3d(
                x=[feature[0]],
                y=[feature[1]],
                z=[feature[2]],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                ),
                text=[str(job_id)[-9:]],
                name=str(job_id)[-9:]
            ))
        elif n_components == 2:
            fig.add_trace(go.Scatter(
                x=[feature[0]],
                y=[feature[1]],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                ),
                text=[str(job_id)[-9:]],
                name=str(job_id)[-9:]
            ))
        else:
            raise ValueError("Invalid number of components. n_components must be 2 or 3.")

    # Add the variance explained by each principal component to the figure's title (if applicable)
    title = f"{n_components}D plot of features ({dim_reduction}~: {explained_variance})" if explained_variance else f"{n_components}D plot of features ({dim_reduction})"
    fig.update_layout(title=title, height=600, width=800)

    # Return the figure
    return fig



def display_timeseries(workflows):
    """
    Display the time series of bytesRead and bytesWritten for a list of workflows using Plotly.
    Args:
        workflows (list of dict): A list of workflows. Each workflow is a dictionary 
                                  with 'bytesRead', 'bytesWritten', and 'timestamp'.
    """
    max_columns = 6
    n = len(workflows)
    n_cols = min(n, max_columns)
    n_rows = math.ceil(n / n_cols)

    fig = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[list(w.keys())[0][-7:] if w is not None else 'None' for w in workflows])

    for i, workflow_data in enumerate(workflows):
        # Skip if workflow_data is None
        if workflow_data is None:
            continue

        workflow_id = list(workflow_data.keys())[0]
        job = workflow_data[workflow_id]

        row = i // n_cols + 1
        col = i % n_cols + 1

        fig.add_trace(go.Scatter(
            x=job['timestamp'],
            y=job['bytesRead'],
            mode='lines',
            name='bytesRead',
            line_color='blue'
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=job['timestamp'],
            y=job['bytesWritten'],
            mode='lines',
            name='bytesWritten',
            line_color='orange'
        ), row=row, col=col)

    fig.update_layout(height=200*n_rows, width=200*n_cols, title_text="Time series for Jobs", showlegend=False)
    return fig




  
# def display_features_3d(features, dim_reduction='pca', seed=42):
#     """
#     Display the features in 3D using PCA, t-SNE or UMAP and Plotly. The closest job to the centroid is highlighted.
#     """
#     features_wo_id = features.loc[:, features.columns != 'job_id']
#     job_ids = features['job_id'].index.values

#     # Perform dimensionality reduction on the features
#     if dim_reduction == 'pca':
#         reducer = PCA(n_components=3, random_state=seed)
#     elif dim_reduction == 'tsne':
#         reducer = TSNE(n_components=3, random_state=seed)
#     elif dim_reduction == 'umap':
#         reducer = umap.UMAP(n_components=3, random_state=seed)
#     else:
#         raise ValueError(f"Invalid dim_reduction value: {dim_reduction}")

#     reduced_features = reducer.fit_transform(features_wo_id)

#     # Calculate the centroid
#     centroid = np.mean(reduced_features, axis=0)

#     # Find the index of the central job
#     distances = euclidean_distances(reduced_features, [centroid])
#     central_idx = np.argmin(distances)

#     # Create a Plotly figure
#     fig = go.Figure()

#     # Add each job's features to the figure as a scatter plot
#     for i, (feature, job_id) in enumerate(zip(reduced_features, job_ids)):
#         color = 'blue'
#         size = 6
#         if i == central_idx:
#             # Highlight the central job
#             color = 'red'
#             size = 8
#         fig.add_trace(go.Scatter3d(
#             x=[feature[0]],
#             y=[feature[1]],
#             z=[feature[2]],
#             mode='markers+text',
#             marker=dict(
#                 size=size,
#                 color=color,
#             ),
#             text=[f'Job {job_id}'],
#             name=f'Job {job_id}'
#         ))

#     # Add the centroid to the figure
#     fig.add_trace(go.Scatter3d(
#         x=[centroid[0]],
#         y=[centroid[1]],
#         z=[centroid[2]],
#         mode='markers',
#         marker=dict(
#             size=8,
#             color='black',
#             opacity=0.5
#         ),
#         name='Centroid'
#     ))

#     # Add the variance explained by each principal component to the figure's title
#     title = f"3D plot of features ({dim_reduction.upper()})"
#     if dim_reduction == 'pca':
#         explained_variance = np.sum(reducer.explained_variance_ratio_)
#         title += f" (explained variance: {explained_variance})"
#     fig.update_layout(title=title)

#     # Return the figure
#     return fig




# def display_timeseries(workflow_searcher, workflow_ids=None):
#     """
#     Display the time series of bytesRead and bytesWritten for a list of workflow IDs using Plotly.
#     If no IDs are provided, all workflows will be plotted.
#     """
#     if workflow_ids is None:
#         workflow_ids = list(workflow_searcher.connector.workflows.keys())

#     n = len(workflow_ids)
#     n_cols = math.ceil(math.sqrt(n))
#     n_rows = math.ceil(n / n_cols)
    
#     fig = sp.make_subplots(rows=n_rows, cols=n_cols)
    
#     for i, workflow_id in enumerate(workflow_ids):
#         row = i // n_cols + 1
#         col = i % n_cols + 1

#         workflow_data = workflow_searcher.extract_workflow_data(workflow_id)
#         job = list(workflow_data.values())[0]

#         fig.add_trace(go.Scatter(
#             x=job['timestamp'],
#             y=job['bytesRead'],
#             mode='lines',
#             name=f'bytesRead_{workflow_id}'
#         ), row=row, col=col)

#         fig.add_trace(go.Scatter(
#             x=job['timestamp'],
#             y=job['bytesWritten'],
#             mode='lines',
#             name=f'bytesWritten_{workflow_id}'
#         ), row=row, col=col)

#     fig.update_layout(height=400*n_rows, width=400*n_cols, title_text="Time series for Jobs")
#     return fig