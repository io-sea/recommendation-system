import os
import json
from loguru import logger
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import plotly.figure_factory as ff
import plotly.graph_objs as go
from datetime import datetime, timedelta

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

        return job2["start_time"] >= job1["start_time"] - (self.threshold * job1_duration) and job2["start_time"] <= job1["start_time"] + (self.threshold * job1_duration)

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
        self.clean_redundancy()
        logger.info("Redundancy removed.")

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


    def dump_graph_to_json(self, file_path=None):
        """
        Dumps the graph with metadata to a JSON file.

        Args:
            file_path (str): Optional. The path to the JSON file to save the graph data.
                             If not provided, it defaults to {self.workflow_folder}_dependencies_job.json.
        """
        if file_path is None:
            # Create the default file name based on the workflow folder name
            wf_folder_name = os.path.basename(os.path.normpath(self.workflow_folder))
            file_path = os.path.join(self.workflow_folder, f"{wf_folder_name}_dependencies_job.json")

        # Convert the graph to a node-link format that is JSON serializable
        graph_data = nx.node_link_data(self.graph)

        # Add the sorted_jobs data to the graph data
        graph_data['sorted_jobs'] = self.sorted_jobs

        # Write the graph data to a JSON file
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=4)

        logger.info(f"Graph data has been dumped to {file_path}")

    def dump_edges_to_json(self, file_path=None):
        """
        Dumps the graph edges with metadata to a JSON file in an edge list format.

        Args:
            file_path (str): Optional. The path to the JSON file to save the edges data.
                             If not provided, it defaults to {self.workflow_folder}_edges.json.
        """
        if file_path is None:
            # Create the default file name based on the workflow folder name
            wf_folder_name = os.path.basename(os.path.normpath(self.workflow_folder))
            file_path = os.path.join(self.workflow_folder, f"{wf_folder_name}_edges.json")

        # Extract the edges and their attributes from the graph
        edges_data = [(u, v, d) for u, v, d in self.graph.edges(data=True)]

        # Write the edges data to a JSON file
        with open(file_path, 'w') as f:
            json.dump(edges_data, f, indent=4)

        logger.info(f"Edges data has been dumped to {file_path}")

    def print_graph(self):
        """
        Prints the graph to the console.
        """

        print("Job Dependency Graph:")
        for node in self.graph.nodes():
            print(f"Job: {node}")
        for edge in self.graph.edges(data=True):
            print(f"From {edge[0]} to {edge[1]} - Type: {edge[2]['type']}")

    def plot_gantt_chart(self):
        """
        Plots a Gantt chart of the jobs using their start and end times.
        """

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create a list for the bar labels and start and end dates
        labels = []
        start_dates = []
        end_dates = []

        # Extract job data for the Gantt chart
        for job_id, job_data in self.sorted_jobs.items():
            labels.append(job_id)
            start_dates.append(datetime.fromtimestamp(job_data['start_time'] / 1000.0))
            end_dates.append(datetime.fromtimestamp(job_data['end_time'] / 1000.0))

        # Create the bars for the Gantt chart
        for i, (start, end) in enumerate(zip(start_dates, end_dates)):
            ax.barh(i, end - start, left=start, height=0.4, align='center')

        # Set the y-axis labels
        ax.set(yticks=range(len(labels)), yticklabels=labels)
        ax.invert_yaxis()  # Invert the y-axis so the first entry is at the top

        # Format the dates on the x-axis
        ax.xaxis_date()  # Tell matplotlib that these are dates
        date_format = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()  # Rotate dates to prevent overlap

        # Draw dependency lines
        for edge in self.graph.edges(data=True):
            start_job = edge[0]
            end_job = edge[1]
            edge_type = edge[2]['type']

            start_index = labels.index(start_job)
            end_index = labels.index(end_job)

            if edge_type == 'sequential':
                ax.annotate('',
                            xy=(datetime.fromtimestamp(self.sorted_jobs[end_job]['start_time'] / 1000.0), end_index),
                            xytext=(datetime.fromtimestamp(self.sorted_jobs[start_job]['end_time'] / 1000.0), start_index),
                            arrowprops=dict(arrowstyle='->', color='r'))
            elif edge_type == 'parallel':
                ax.annotate('',
                            xy=(datetime.fromtimestamp(self.sorted_jobs[start_job]['start_time'] / 1000.0), start_index),
                            xytext=(datetime.fromtimestamp(self.sorted_jobs[end_job]['start_time'] / 1000.0), end_index),
                            arrowprops=dict(arrowstyle='-', color='g', linestyle='dotted'))
            elif edge_type == 'delay':
                # For delay, we assume the arrow starts a bit after the end of job1 and points to the start of job2
                ax.annotate('',
                            xy=(datetime.fromtimestamp(self.sorted_jobs[end_job]['start_time'] / 1000.0), end_index),
                            xytext=(datetime.fromtimestamp(self.sorted_jobs[start_job]['end_time'] / 1000.0 + self.threshold * (self.sorted_jobs[start_job]['end_time'] - self.sorted_jobs[start_job]['start_time']) / 1000.0), start_index),
                            arrowprops=dict(arrowstyle='-', color='b', linestyle='dashed'))

        return plt


    def create_gantt_chart(self, fig_size=(1024, 768), show_grid=True):
        """
        Creates a Gantt chart using Plotly with curved dependency arrows.

        Args:
            fig_size (tuple): Figure size in the form of (width, height).
            show_grid (bool): Whether to show grid lines in the chart.

        Returns:
            plotly.graph_objs._figure.Figure: The Plotly figure object for the Gantt chart.
        """
        df = []
        for job_id, job_data in self.sorted_jobs.items():
            df.append(dict(Task=job_id, Start=self._ms_to_datetime(job_data['start_time']),
                        Finish=self._ms_to_datetime(job_data['end_time']), Resource='Job'))

        fig = ff.create_gantt(df, colors=['rgb(210, 210, 210)'], index_col='Resource', show_colorbar=False,
                            bar_width=0.4, showgrid_x=show_grid, showgrid_y=show_grid)

        annotations = []
        for edge in self.graph.edges(data=True):
            start_job = edge[0]
            end_job = edge[1]
            edge_type = edge[2]['type']

            # Calculate positions for start and end
            y_start = df.index(next(item for item in df if item["Task"] == start_job))
            y_end = df.index(next(item for item in df if item["Task"] == end_job))
            x_start = self.sorted_jobs[start_job]['end_time']
            x_end = self.sorted_jobs[end_job]['start_time']

            # Add arrows with annotations
            annotations.append(
                dict(
                    ax=self._ms_to_datetime(x_start), ay=y_start, axref='x', ayref='y',
                    x=self._ms_to_datetime(x_end), y=y_end, xref='x', yref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='green' if edge_type == 'parallel' else 'red'
                )
            )

        fig.update_layout(annotations=annotations)
        fig.update_layout(
            height=fig_size[1],
            width=fig_size[0],
            title='Job Scheduling Gantt Chart',
            xaxis_title='Time',
            yaxis_title='Jobs',
            hovermode='closest',
            yaxis=dict(autorange='reversed')  # Reverse the y-axis to have jobs from top to bottom
        )

        return fig

    @staticmethod
    def _ms_to_datetime(milliseconds):
        """Convert milliseconds since epoch to datetime."""
        return datetime.fromtimestamp(milliseconds / 1000)
