from loguru import logger
import os
import json
from collections import defaultdict

class JobDependencyAnalyzer:
    """
    A class to analyze job dependencies based on job metadata.

    Attributes:
        workflow_folder (str): The path to the workflow folder containing jobs.
        threshold (float): A threshold for determining job dependencies.
        sorted_jobs (dict): A dictionary containing job IDs as keys and their start and end times as values.
        dependencies (dict): A dictionary containing lists of job dependencies of different types (sequential, parallel, delay).
    """

    def __init__(self, workflow_folder, threshold=0.1):
        """
        Initializes the JobDependencyAnalyzer object.

        Args:
            workflow_folder (str): The path to the folder containing job data.
            threshold (float): The threshold for determining job dependencies.
        """
        self.workflow_folder = workflow_folder
        self.threshold = threshold
        self.sorted_jobs = self.extract_and_sort_jobs()
        self.dependencies = defaultdict(list)

    def is_file_extension(self, filename, extension):
        """
        Checks if a file has a specific extension.

        Args:
            filename (str): The name of the file.
            extension (str): The file extension to check.

        Returns:
            bool: True if the file has the specified extension, False otherwise.
        """
        return filename.endswith(f'.{extension}')

    def is_connected(self, start_job, end_job, visited=None, path_cache=None,
                     dependency_type=None):
        """
        Checks if two jobs are connected directly or indirectly via sequential or parallel dependencies.

        Args:
            start_job (str): The starting job ID.
            end_job (str): The ending job ID.
            visited (set): A set to keep track of visited jobs.

        Returns:
            bool: True if the jobs are connected, False otherwise.
        """
        if visited is None:
            visited = set()

        if path_cache is None:
            path_cache = {}

        # Check cache first to avoid redundant calculations
        if (start_job, end_job) in path_cache:
            return path_cache[(start_job, end_job)]

        # Mark the current node as visited
        visited.add(start_job)

        logger.debug(f"Checking connectivity from {start_job} to {end_job}")
        logger.debug(f"Visited so far: {visited}")

        # If the end node is reached
        if start_job == end_job:
            return True

        dependency_types = [dependency_type] if dependency_type else ['sequential', 'parallel', 'delay']
        for dep_type in dependency_types:
            for next_job in self.dependencies.get(dep_type, []):
                from_job, to_job = next_job[0], next_job[1]

                # If a relevant connection is found and the next job hasn't been visited
                if start_job == from_job and to_job not in visited:
                    if self.is_connected(to_job, end_job, visited):
                        logger.debug(f"Job {start_job} is connected to {end_job} via {to_job}")
                        return True

        # If no connection is found, update the cache with the result
        path_cache[(start_job, end_job)] = False
        logger.debug(f"Job {start_job} is not connected to {end_job}")
        return False

    def extract_and_sort_jobs(self):
        """
        Extracts job metadata from the workflow folder and sorts them based on their start times.

        The method navigates through each sub-folder within the provided workflow folder, each corresponding to a job.
        Within each job folder, it looks for a file named 'volume.json' that contains the job's timing metadata.
        The start and end timestamps are extracted from this JSON file.

        Once all the job metadata is collected, the jobs are sorted by their start times for further dependency analysis.

        Attributes Set:
            sorted_jobs (dict): A dictionary containing sorted job data. Each key is a job ID, and the value is another
                                dictionary containing 'start_time' and 'end_time' for that job.

        Example structure of sorted_jobs:
        {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 1500, 'end_time': 2500},
            ...
        }

        Steps:
        1. Initialize an empty dictionary `sorted_jobs` to hold job data.
        2. Loop through all the sub-folders in the `workflow_folder`.
        3. For each sub-folder, check if a 'volume.json' file exists.
        4. If it exists, read the file and extract the start and end timestamps.
        5. Store these timestamps in the `sorted_jobs` dictionary.
        6. Sort the `sorted_jobs` dictionary by the start times of the jobs.
        """
        # Initialize dictionary to hold job data
        self.sorted_jobs = defaultdict(dict)

        # Loop through all job folders in the workflow folder
        for job_folder in os.listdir(self.workflow_folder):
            job_folder_path = os.path.join(self.workflow_folder, job_folder)

            # Ensure that we are dealing with job folder
            if os.path.isdir(job_folder_path):
                json_file_path = os.path.join(job_folder_path, 'volume.json')

                # Check if volume.json exists in the job folder
                if os.path.exists(json_file_path):
                    # Read the JSON file into a list of lists
                    with open(json_file_path, 'r') as f:
                        volume_data = json.load(f)

                    # Extract the job start and end timestamps
                    start_time = volume_data[0][0]
                    end_time = volume_data[-1][0]

                    # Store data in the dictionary
                    self.sorted_jobs[job_folder] = {'start_time': start_time, 'end_time': end_time}

        # Sort jobs by their start_time
        self.sorted_jobs = dict(sorted(self.sorted_jobs.items(), key=lambda x: x[1]['start_time']))

    def is_sequential(self, job1, job2, threshold):
        """
        Check if job2 is sequential to job1 within a given threshold.

        Args:
        - job1 (dict): Dictionary containing 'start_time' and 'end_time' for job1.
        - job2 (dict): Dictionary containing 'start_time' and 'end_time' for job2.
        - threshold (float): The threshold value for deciding if the jobs are sequential.

        Returns:
        - bool: True if job2 is sequential to job1, False otherwise.
        """
        job1_duration = job1['end_time'] - job1['start_time']
        return job2['start_time'] in range(int(job1['end_time'] - self.threshold * job1_duration), int(job1['end_time'] + self.threshold * job1_duration))

    def is_parallel(self, job1, job2, threshold):
        """
        Check if job1 and job2 are parallel within a given threshold.

        Args:
        - job1 (dict): Dictionary containing 'start_time' and 'end_time' for job1.
        - job2 (dict): Dictionary containing 'start_time' and 'end_time' for job2.
        - threshold (float): The threshold value for deciding if the jobs are parallel.

        Returns:
        - bool: True if job1 and job2 are parallel, False otherwise.
        """
        job1_duration = job1['end_time'] - job1['start_time']
        return job2['start_time'] in range(int(job1['start_time'] - threshold * job1_duration), int(job1['start_time'] + threshold * job1_duration))

    def remove_redundant_delays(self):
        """
        Removes redundant delay entries from the dependencies dictionary.
        """
        # Initialize cache for is_connected method
        path_cache = {}

        # Initialize a new list to hold only the necessary delay dependencies
        necessary_delays = []

        # Sort the delays by the delay time, descending
        sorted_delays = sorted(self.dependencies['delay'], key=lambda x: x[2], reverse=True)

        for delay in sorted_delays:
            job1, job2, _ = delay
            # Temporarily remove the delay entry
            self.dependencies['delay'].remove(delay)
            # Check if a path still exists without this delay
            if not self.is_connected(job1, job2, path_cache=path_cache):
                # If no path exists, the delay is necessary
                necessary_delays.append(delay)
            # Add the delay back if it was deemed redundant
            else:
                self.dependencies['delay'].append(delay)

        # Update the 'delay' dependencies with only necessary ones
        self.dependencies['delay'] = necessary_delays

    def remove_redundant_parallel(self):
        """
        Removes parallel entries that are already connected via sequential dependencies.
        """
        # Initialize a new list to hold non-redundant parallel dependencies
        non_redundant_parallel = []

        # Iterate through each parallel dependency
        for parallel_entry in self.dependencies['parallel']:
            job1, job2 = parallel_entry

            # Check if a sequential path exists between job1 and job2
            if not self.is_connected(job1, job2, dependency_type='sequential'):
                # If no sequential path exists, then the parallel dependency is necessary
                non_redundant_parallel.append(parallel_entry)

        # Update the 'parallel' dependencies with non-redundant ones
        self.dependencies['parallel'] = non_redundant_parallel

        # Optionally, log the updated dependencies
        logger.info(f"Updated parallel dependencies: {self.dependencies['parallel']}")

    def analyze_dependencies(self):
        """
        Analyze job dependencies based on job timings.

        """
        # Initialize a dictionary to hold dependencies
        self.dependencies = {
            'sequential': [],
            'parallel': [],
            'delay': []
        }

        job_ids = list(self.sorted_jobs.keys())

        for i in range(len(job_ids)):
            job1_id = job_ids[i]
            job1 = self.sorted_jobs[job1_id]

            for j in range(i + 1, len(job_ids)):
                job2_id = job_ids[j]
                job2 = self.sorted_jobs[job2_id]

                if self.is_sequential(job1, job2, self.threshold):
                    self.dependencies['sequential'].append([job1_id, job2_id])
                    logger.info(f"Job {job1_id} and {job2_id} are sequential : {self.is_sequential(job1, job2, self.threshold)}")
                elif self.is_parallel(job1, job2, self.threshold):
                    self.dependencies['parallel'].append([job1_id, job2_id])

        # Remove redundant parallel dependencies
        self.remove_redundant_parallel()

        # Second pass to determine delay relationships
        for i in range(len(job_ids)):
            job1_id = job_ids[i]
            job1 = self.sorted_jobs[job1_id]

            for j in range(i + 1, len(job_ids)):
                job2_id = job_ids[j]
                job2 = self.sorted_jobs[job2_id]

                # Check if job1 and job2 are already connected via some other path
                if not self.is_connected(job1_id, job2_id):
                    logger.debug(f"Job {job1_id} and {job2_id} are not connected : {self.is_connected(job1_id, job2_id)}")
                    delay = job2['start_time'] - job1['end_time']
                    self.dependencies['delay'].append([job1_id, job2_id, delay])

        # After populating self.dependencies in analyze_dependencies
        logger.debug(f"Analyzed dependencies with redundancy: {self.dependencies}")

        # Call the method to remove redundant delays and reorder them
        self.remove_redundant_delays()

        self.dependencies["delay"]
        logger.info(f"Analyzed dependencies: {self.dependencies}")

    def save_to_json(self, file_path):
        """
        Save dependencies to a JSON file.

        Parameters:
            file_path (str): The path where the JSON file will be saved.
        """
        with open(file_path, 'w') as f:
            json.dump(self.dependencies, f)

        logger.info(f"Dependencies saved to {file_path}")
