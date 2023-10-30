from loguru import logger
import os
import json
from collections import defaultdict

class JobDependencyAnalyzer:
    """A class to analyze job dependencies based on job metadata."""

    def __init__(self, workflow_folder):
        """
        Initialize the JobDependencyAnalyzer class.

        Parameters:
            workflow_folder (str): The folder path containing job metadata.
        """
        self.workflow_folder = workflow_folder
        self.sorted_jobs = None
        self.dependencies = defaultdict(list)

    def is_file_extension(self, filename, extension):
        """Check if a file has a specific extension."""
        return filename.endswith(f'.{extension}')

    def extract_and_sort_jobs(self):
        """
        Extract and sort job metadata.

        This method reads job metadata from JSON files within the workflow folder,
        extracts the start and end timestamps for each job, and sorts the jobs by their start timestamps.
        """
        logger.info("Extracting and sorting jobs...")

        # Initialize dictionary to hold job data
        workflow_data = defaultdict(list)

        # Loop through all job folders in the workflow folder
        for job_folder in os.listdir(self.workflow_folder):
            job_folder_path = os.path.join(self.workflow_folder, job_folder)

            # Check if the item is a directory (i.e., a job folder)
            if os.path.isdir(job_folder_path):
                json_file_path = os.path.join(job_folder_path, 'volume.json')

                # Check if volume.json exists in the job folder
                if os.path.exists(json_file_path):
                    logger.debug(f"Processing volume.json for job {job_folder}...")

                    # Read the JSON file into a list of lists
                    with open(json_file_path, 'r') as f:
                        volume_data = json.load(f)

                    # Extract the job start and end timestamps
                    start_time = volume_data[0][0]
                    end_time = volume_data[-1][0]

                    # Store data in the dictionary
                    workflow_data[job_folder] = {'start_time': start_time, 'end_time': end_time}

        # Sort jobs by start_time
        self.sorted_jobs = dict(sorted(workflow_data.items(), key=lambda item: item[1]['start_time']))

    def is_sequential(self, job1, job2, threshold=0.1):
        job1_duration = job1['end_time'] - job1['start_time']

        return job1['end_time'] in range(int(job2['start_time'] - threshold * job1_duration), int(job2['start_time'] + threshold * job1_duration))

    def is_parallel(self, job1, job2, threshold=0.1):
        job1_duration = job1['end_time'] - job1['start_time']
        return job1['start_time'] in range(int(job2['start_time'] - threshold * job1_duration), int(job2['start_time'] + threshold * job1_duration))

    def is_delay(self, job1, job2, threshold=0.1):
        job1_duration = job1['end_time'] - job1['start_time']
        return job2['start_time'] > job1['end_time'] + threshold * job1_duration

    def analyze_dependencies(self):
        """
        Analyze job dependencies based on sorted job metadata.
        """
        logger.info("Analyzing job dependencies...")

        for job_id1, job1 in self.sorted_jobs.items():
            for job_id2, job2 in self.sorted_jobs.items():
                if job_id1 == job_id2:
                    continue

                if self.is_sequential(job1, job2):
                    self.dependencies['sequential'].append([job_id1, job_id2])

                if self.is_parallel(job1, job2):
                    self.dependencies['parallel'].append([job_id1, job_id2])

                if self.is_delay(job1, job2):
                    delay = job2['start_time'] - job1['end_time']
                    self.dependencies['delay'].append([job_id1, job_id2, delay])

    def save_to_json(self, file_path):
        """
        Save dependencies to a JSON file.

        Parameters:
            file_path (str): The path where the JSON file will be saved.
        """
        with open(file_path, 'w') as f:
            json.dump(self.dependencies, f)

        logger.info(f"Dependencies saved to {file_path}")
