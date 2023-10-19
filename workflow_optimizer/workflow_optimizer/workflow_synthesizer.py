"""
Workflow Synthesizer Module
===========================

Module Overview
---------------
This module provides functionalities for synthesizing workflow representations
from time-series data. The objective is to facilitate the study, simulation,
and optimization of HPC workflows, particularly in the context of data placement
strategies.

Key Features
------------
- Time-series Data Aggregation
- Data Imputation
- Workflow Visualization
- Data Export
- Extensibility

Usage
-----
Example:
```python
from workflow_synthesizer import WorkflowSynthesizer

# Initialize synthesizer
synthesizer = WorkflowSynthesizer()

# Synthesize workflow
synthesizer.synthesize(data_for_jobs)

# Export workflow
synthesizer.export_to_json("output.json")

# Plot workflow
synthesizer.plot_workflow()

Dependencies
    Pandas
    NumPy
    Matplotlib
.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

class WorkflowSynthesizer:
    """
    Class for synthesizing a workflow from a dictionary of jobs.
    The jobs and the workflow are represented as dataframes with time-series data.
    """

    def __init__(self):
        """
        Initialize a new instance of the WorkflowSynthesizer class.
        """
        self.workflow = pd.DataFrame()

    def synthesize(self, data_for_jobs):
        """
        Synthesize a workflow from the provided jobs.

        Args:
            data_for_jobs (dict): A dictionary where keys are job names
            and values are DataFrames with columns 'timestamp', 'bytesRead',
            and 'bytesWritten'.
        """
        min_timestamp = min(df['timestamp'].min() for df in
                            data_for_jobs.values())
        max_timestamp = max(df['timestamp'].max() for df in
                            data_for_jobs.values())

        # Create a DataFrame with a uniform timestamp range
        synthetic_timestamps = np.arange(min_timestamp, max_timestamp + 1,
                                         5000)
        synthetic_df = pd.DataFrame({'timestamp': synthetic_timestamps})

        for job_name, job_df in data_for_jobs.items():
            # Merge with the synthetic DataFrame to get uniform timestamps
            merged_df = pd.merge(synthetic_df, job_df, on='timestamp', how='left').fillna(0)

            # Sum up the bytesRead and bytesWritten across all jobs
            if self.workflow.empty:
                self.workflow = merged_df
            else:
                self.workflow['bytesRead'] += merged_df['bytesRead']
                self.workflow['bytesWritten'] += merged_df['bytesWritten']

    def to_dict(self):
        """
        Convert the synthesized workflow to a dictionary format.

        Returns:
            dict: The synthesized workflow in dictionary format.
        """
        output = {}
        output['timestamp'] = self.workflow['timestamp'].tolist()
        output['bytesRead'] = self.workflow['bytesRead'].tolist()
        output['bytesWritten'] = self.workflow['bytesWritten'].tolist()
        return output

    def plot_workflow(self):
        """
        Plot the time series of bytesRead and bytesWritten for the synthesized workflow.
        """
        plt.figure(figsize=(14, 6))
        plt.plot(self.workflow['timestamp'], self.workflow['bytesRead'], label='Bytes Read')
        plt.plot(self.workflow['timestamp'], self.workflow['bytesWritten'], label='Bytes Written')
        plt.xlabel('Timestamp')
        plt.ylabel('Bytes')
        plt.title('Bytes Read and Written Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
