
from workflow_optimizer.utils import (is_file_extension, get_column_names,
                                      camel_case_to_snake_case, list_and_classify_directory_contents)
from app_decomposer.job_decomposer import ComplexDecomposer
from app_decomposer.config_parser import Configuration
from unittest.mock import patch
import os
import pandas as pd
from loguru import logger
import json
from pathlib import Path
import numpy as np

# Define the base folder path as a constant
BASE_WF_FOLDER = '/home_nfs/mimounis/iosea-wp3-recommandation-system/dataset_generation/dataset_deep/'


class WorkflowDecomposer:
    """
    Class for decomposing workflows into manageable components.

    Attributes:
        v0_threshold (float): The threshold value used in the job decomposition.
        wf_folder (str): The workflow folder.

    Methods:
        get_job_timeseries_from_json_file: Extracts time series data for a specific job from JSON files.
        decompose_ioi_job: Decomposes a job within a workflow.
        decompose_workflow: Decomposes all jobs in a workflow.
    """

    def __init__(self, v0_threshold=0.01,
                 workflow_name='ECMWF-649c3c40cc9340246f87cb58'):
        """
        Initializes the WorkflowDecomposer with a specified threshold and optional workflow name.

        Parameters:
            v0_threshold (float): The threshold for job decomposition. Default is 0.01.
            workflow_name (str): The optional workflow folder name. Default is an empty string.
        """
        self.v0_threshold = v0_threshold
        self.wf_folder = os.path.join(BASE_WF_FOLDER, workflow_name)
        logger.info(f"WorkflowDecomposer initialized with threshold {self.v0_threshold} and workflow folder {self.wf_folder}")

    def get_job_timeseries_from_json_file(self, job_id, skip_columns=[]):
        """
        Extracts time series data for a specific job within a workflow folder.

        Parameters:
            wf_folder (str): The workflow folder containing job data.
            job_id (str): The specific job ID to extract time series data for.
            skip_columns (list): List of columns to be skipped.

        Returns:
            dict: A dictionary containing time series data for the job.
        """
        job_folder_path = os.path.join(self.wf_folder, str(job_id))

        if not os.path.exists(job_folder_path):
            logger.error(f"Job folder for job_id {job_id} not found in {self.wf_folder}")
            raise FileNotFoundError(f"Job folder for job_id {job_id} not found in {self.wf_folder}")

        logger.info(f"Processing job_id {job_id} from folder {self.wf_folder}")
        # Initialize a dictionary to hold time series data
        job_timeseries = {}

        # Mapping of column names to final output names
        column_name_map = {
            "bytesRead": "bytesRead",
            "bytesWritten": "bytesWritten",
            "operations_count_read": "operationRead",
            "operations_count_write": "operationWrite",
            "access_pattern_read_random": "accessRandRead",
            "access_pattern_read_sequential": "accessSeqRead",
            "access_pattern_read_stride": "accessStrRead",
            "access_pattern_read_unclassified": "accessUnclRead",
            "access_pattern_write_random": "accessRandWrite",
            "access_pattern_write_sequential": "accessSeqWrite",
            "access_pattern_write_stride": "accessStrWrite",
            "access_pattern_write_unclassified": "accessUnclWrite",
        }

        # Mapping of JSON file names to output dictionary keys
        json_key_map = {
            "volume.json": "volume",
            "operationsCount.json": "operationsCount",
            "accessPatternRead.json": "accessPattern",
            "accessPatternWrite.json": "accessPattern",
            "ioSizesRead.json": "ioSizes",
            "ioSizesWrite.json": "ioSizes"
        }

        for json_file in os.listdir(job_folder_path):
            if is_file_extension(json_file, "json"):
                json_file_path = os.path.join(job_folder_path, json_file)
                df = pd.read_json(json_file_path)
                column_names = get_column_names(json_file)
                df.columns = column_names
                df_clean = df.drop_duplicates(subset=['timestamp'])

                key = json_key_map.get(json_file, json_file)
                if key not in job_timeseries:
                    job_timeseries[key] = {}

                for column in df_clean.columns:
                    new_col_name = column_name_map.get(column, column)
                    if new_col_name not in skip_columns:
                        job_timeseries[key][new_col_name] = df_clean[column].to_numpy()
        return job_timeseries

    def process_job_folder(self, job_folder, save_to_csv=None):
        """
        Processes a specific job folder within a workflow directory.

        Parameters:
            job_folder (str): The specific job folder to process.
            save_to_csv (bool): Flag to determine if the DataFrame should be saved as CSV. Default is None.

        Returns:
            pd.DataFrame: A DataFrame containing the processed data for the specific job.
            dict: A dictionary containing the processed data for all jobs.
        """
        logger.info(f"----\nProcessing workflow: {self.wf_folder}...")
        logger.info(f"Found job_folder: {job_folder}")

        # Initialize an empty DataFrame to hold data for this job
        df_for_this_job = pd.DataFrame()

        # Initialize an empty dictionary to hold data for all jobs
        data_for_jobs = {}

        # Loop through all JSON files in the job folder
        for json_file in os.listdir(os.path.join(self.wf_folder, job_folder)):
            if is_file_extension(json_file, "json"):
                logger.info(f"Processing {json_file}...")

                # Construct the full path to the JSON file
                json_file_path = os.path.join(self.wf_folder, job_folder,
                                              json_file)

                # Read the JSON file into a list of dictionaries
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)

                # Create a temporary DataFrame from the JSON data
                df_temp = pd.DataFrame(json_data, columns=get_column_names(json_file))

                # Merge the temporary DataFrame into the DataFrame for this job
                if df_for_this_job.empty:
                    df_for_this_job = df_temp
                else:
                    df_for_this_job = pd.merge(df_for_this_job, df_temp, on='timestamp', how='outer')

                # Optionally save the DataFrame for this job to a CSV file
                if save_to_csv:
                    csv_file_path = os.path.join(self.wf_folder, f"{job_folder}.csv")
                    logger.info(f"Saving to {csv_file_path}")
                    df_for_this_job.to_csv(csv_file_path, index=False)

                # Add this DataFrame to the dictionary
                data_for_jobs[job_folder] = df_for_this_job

        return data_for_jobs

    def decompose_ioi_job(self, job_id):
        """
        Decomposes a specific job within a workflow.

        Parameters:
            wf_folder (str): The workflow folder containing job data.
            job_id (str): The job ID to decompose.

        Returns:
            tuple: A tuple containing the job representation and its phase features.
        """
        logger.info(f"Decomposing job {job_id} in workflow folder {self.wf_folder}")
        with patch.object(ComplexDecomposer, 'get_job_timeseries') as mock_get_timeseries:
            with patch.object(Configuration, 'get_kc_token') as mock_get_kc_token:
                with patch.object(ComplexDecomposer, 'get_job_node_count') as mock_get_node_count:
                    mock_get_timeseries.return_value = self.get_job_timeseries_from_json_file(job_id)
                    mock_get_kc_token.return_value = 'token'
                    mock_get_node_count.return_value = 1

                    cd = ComplexDecomposer(v0_threshold=self.v0_threshold)
                    representation = cd.get_job_representation(merge_clusters=True)
                    phase_features = cd.get_phases_features(representation, job_id=job_id)
                    return representation, phase_features

    def save_representation(self, job_id, representation):
        """
        Save the job's representation data as a JSON file.

        Parameters:
            job_id (str): The job ID whose representation is to be saved.
            representation (dict): The representation data.
        """
        # Function to convert NumPy data types to Python native types
        def default_conversion(o):
            if isinstance(o, np.int64):
                return int(o)
            raise TypeError

        # Create a folder to store decomposed data if it doesn't exist
        decomposed_data_folder = os.path.join(self.wf_folder, 'decomposed_data')
        if not os.path.exists(decomposed_data_folder):
            os.mkdir(decomposed_data_folder)

        # Define the file path
        file_path = os.path.join(decomposed_data_folder, f"{job_id}_representation.json")

        # Write the representation data to a JSON file
        with open(file_path, 'w') as f:
            json.dump(representation, f, default=default_conversion)

        logger.info(f"Saved representation for job {job_id} to {file_path}")

    def save_phase_features(self, job_id, phase_features):
        """
        Save the job's phase features as a JSON file.

        Parameters:
            job_id (str): The job ID whose phase features are to be saved.
            phase_features (dict): The phase features data.
        """
        # Function to convert NumPy data types to Python native types
        def default_conversion(o):
            if isinstance(o, np.int64):
                return int(o)
            raise TypeError

        # Create a folder to store decomposed data if it doesn't exist
        decomposed_data_folder = os.path.join(self.wf_folder, 'decomposed_data')
        if not os.path.exists(decomposed_data_folder):
            os.mkdir(decomposed_data_folder)

        # Define the file path
        file_path = os.path.join(decomposed_data_folder, f"{job_id}_phase_features.json")

        # Write the phase features data to a JSON file
        with open(file_path, 'w') as f:
            json.dump(phase_features, f, default=default_conversion)

        logger.info(f"Saved phase features for job {job_id} to {file_path}")

    def decompose_workflow(self):
        """
        Decomposes all jobs in a given workflow folder.

        Parameters:
            wf_folder (str): The workflow folder containing job data.

        Returns:
            dict: A dictionary containing decomposed data for all jobs.
        """
        logger.info(f"Decomposing all jobs in workflow folder {self.wf_folder}")

        # Initialize a dictionary to store job representations and phase features
        jobs_representations = {}

        # Loop through all folders in the workflow directory
        for job_folder in os.listdir(self.wf_folder):

            # Check if the folder is a directory and its name is numeric
            if os.path.isdir(os.path.join(self.wf_folder, job_folder)) and job_folder.isdigit():

                # Process the job folder and get a DataFrame for this job
                data_for_jobs = self.process_job_folder(job_folder)

                # Loop through the DataFrame dictionary returned by process_job_folder
                for job_id, df_for_this_job in data_for_jobs.items():

                    # Check if the DataFrame has more than one row
                    if df_for_this_job.shape[0] > 1:

                        # Decompose the job and get its representation and phase features
                        representation, phase_features = self.decompose_ioi_job(job_id)

                        # Save the representation and phase features
                        self.save_representation(job_id, representation)
                        self.save_phase_features(job_id, phase_features)

                        # Log the representation
                        logger.info(f"{job_id} representation {representation}")

                        # Store the representation and phase features in the dictionary
                        jobs_representations[job_id] = {'representation': representation, 'phase_features': phase_features}

        return jobs_representations


# Example usage
wd = WorkflowDecomposer(workflow_name="LQCD-64873bafcc9340246f412faf")
#wf_folder = "LQCD-64873bafcc9340246f412faf"
#representation, phase_features = wd.decompose_ioi_job('371912')
decomposed_data = wd.decompose_workflow()

