import os
import json
from loguru import logger
import networkx as nx

class JobDependencyAnalyzerNX:
    def __init__(self, workflow_folder, threshold=0.1):
        self.workflow_folder = workflow_folder
        self.threshold = threshold
        self.graph = nx.DiGraph()
        self.sorted_jobs = self.extract_and_sort_jobs()

    def extract_and_sort_jobs(self):
        """
        Extracts job metadata from the workflow folder and sorts them based on their start times.

        The method navigates through each sub-folder within the provided workflow folder, each corresponding to a job.
        Within each job folder, it looks for a file named 'volume.json' that contains the job's timing metadata.
        The start and end timestamps are extracted from this JSON file.

        Once all the job metadata is collected, the jobs are sorted by their start times for further dependency analysis.

        Returns:
            dict: A dictionary with job IDs as keys and another dictionary containing 'start_time' and 'end_time' as values.
        """
        sorted_jobs = {}
        for job_folder in os.listdir(self.workflow_folder):
            job_folder_path = os.path.join(self.workflow_folder, job_folder)
            if os.path.isdir(job_folder_path):
                json_file_path = os.path.join(job_folder_path, 'volume.json')
                if os.path.exists(json_file_path):
                    with open(json_file_path, 'r') as f:
                        volume_data = json.load(f)
                    start_time, end_time = volume_data[0][0], volume_data[-1][0]
                    sorted_jobs[job_folder] = {'start_time': start_time, 'end_time': end_time}
        sorted_jobs = dict(sorted(sorted_jobs.items(), key=lambda x: x[1]['start_time']))
        logger.info("Jobs extracted and sorted by start time.")
        return sorted_jobs

    def is_sequential(self, job1, job2):
        """
        Check if job2 starts after job1 finishes within the threshold.

        Args:
            job1 (dict): Metadata for job1.
            job2 (dict): Metadata for job2.

        Returns:
            bool: True if job2 is sequential to job1, False otherwise.
        """
        job1_end = job1['end_time']
        job2_start = job2['start_time']
        job1_duration = job1['end_time'] - job1['start_time']
        return job2_start >= job1_end - (self.threshold * job1_duration) and job2_start <= job1_end + (self.threshold * job1_duration)


    def is_parallel(self, job1, job2):
        """
        Check if job1 and job2 are running in parallel within the threshold.

        Args:
            job1 (dict): Metadata for job1.
            job2 (dict): Metadata for job2.

        Returns:
            bool: True if job1 and job2 are parallel, False otherwise.
        """

        job1_duration = job1['end_time'] - job1['start_time']
        return job2["start_time"] >= job1["start_time"] - (self.threshold * job1_duration) and job2["start_time"] >= job1["start_time"] - (self.threshold * job1_duration)

    def analyze_dependencies(self):
        """
        Analyze job dependencies based on job timings and create graph edges.

        Sequential edges are added if job2 starts after job1 within a threshold.
        Parallel edges are added if job1 and job2 are running at the same time.
        Delay edges are added if neither sequential nor parallel relationships are found.
        """
        job_ids = list(self.sorted_jobs.keys())

        for i in range(len(job_ids)):
            job1_id = job_ids[i]
            job1 = self.sorted_jobs[job1_id]

            for j in range(i + 1, len(job_ids)):
                job2_id = job_ids[j]
                job2 = self.sorted_jobs[job2_id]

                if self.is_sequential(job1, job2):
                    self.graph.add_edge(job1_id, job2_id, type='sequential')
                elif self.is_parallel(job1, job2):
                    self.graph.add_edge(job1_id, job2_id, type='parallel')
                else:
                    # Calculate delay and add as an edge attribute
                    delay = job2['start_time'] - job1['start_time']
                    self.graph.add_edge(job1_id, job2_id, type='delay', delay=delay)

        logger.info("Dependencies analyzed and graph populated.")

    def find_nodes_with_multiple_incoming_edges(self, edge_type=None):
        """
        Finds nodes that have multiple incoming edges, with an optional filter for edge type.

        Args:
            edge_type (str, optional): The type of edges to consider ('sequential', 'parallel', 'delay'). If None, all edge types are considered.

        Returns:
            A list of tuples, where each tuple contains a node and the number of incoming edges of the specified type.
        """
        logger.debug("Finding nodes with multiple incoming edges.")
        nodes_with_multiple_incoming_edges = []

        for node in self.graph.nodes():
            # Get all incoming edges for the node
            incoming_edges = self.graph.in_edges(node, data=True)
            # If edge_type is specified, filter the incoming edges by type
            if edge_type is not None:
                incoming_edges = [edge for edge in incoming_edges if edge[2].get('type') == edge_type]
            # Check if there are multiple incoming edges of the specified type
            if len(incoming_edges) > 1:
                nodes_with_multiple_incoming_edges.append((node, len(incoming_edges)))

        logger.debug(f"Nodes with multiple incoming edges: {nodes_with_multiple_incoming_edges}")
        return nodes_with_multiple_incoming_edges


    def remove_edges_by_priority(self, node):
        """
        Remove incoming edges to a node by priority: delay, parallel, then sequential.
        Only one edge of the highest priority will remain.

        Args:
            node (str): The node for which to remove incoming edges.
        """
        # Define a priority for each edge type
        priorities = {'delay': 1, 'parallel': 2, 'sequential': 3}

        # Collect all incoming edges for the given node
        incoming_edges = list(self.graph.in_edges(node, data=True))

        # If there is only one or no incoming edge, there's nothing to do
        if len(incoming_edges) <= 1:
            return

        # Sort the edges by their priority
        incoming_edges.sort(key=lambda edge: priorities[edge[2].get('type', '')])

        # Keep the highest priority edge and remove the others
        for edge in incoming_edges[:-1]:  # Skip the last edge, which is the highest priority edge
            self.graph.remove_edge(edge[0], node)
            logger.debug(f"Removed edge from '{edge[0]}' to '{node}' of type '{edge[2].get('type')}'")

    def clean_redundancy(self):
        """
        Cleans redundant edges in the graph by finding nodes with multiple incoming edges and removing
        the less prioritized edges. It keeps only the highest priority edge based on the defined
        priorities: delay, parallel, then sequential.

        This method assumes that `find_nodes_with_multiple_incoming_edges` and
        `remove_edges_by_priority` have been defined and are available for use.
        """
        # Find nodes with multiple incoming edges
        nodes_with_redundancies = self.find_nodes_with_multiple_incoming_edges()

        # Loop through each node and clean up redundancies
        for node, count in nodes_with_redundancies:
            logger.debug(f"Cleaning redundancies for node: {node}")
            self.remove_edges_by_priority(node)

        logger.info("Redundancy cleaning complete.")







