import os
import json
import unittest
from loguru import logger
from unittest.mock import patch, mock_open
from workflow_optimizer.job_dependency_analyzer import JobDependencyAnalyzer



class TestJobDependencyAnalyzer(unittest.TestCase):
    """
    Unit tests for the JobDependencyAnalyzer class.
    """

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_extract_and_sort_jobs(self, mock_file, mock_exists, mock_isdir, mock_listdir):
        """
        Test the extract_and_sort_jobs method.
        """

        # Mocking behavior
        mock_listdir.return_value = ['job1', 'job2']
        mock_isdir.return_value = True
        mock_exists.return_value = True

        # Sample job data
        mock_file().read.return_value = json.dumps([
            [1000, 10],
            [2000, 20]
        ])

        # Initialize object
        analyzer = JobDependencyAnalyzer("/some/path")

        # Run method
        analyzer.extract_and_sort_jobs()

        # Validate output
        self.assertEqual(analyzer.sorted_jobs, {'job1': {'start_time': 1000, 'end_time': 2000}, 'job2': {'start_time': 1000, 'end_time': 2000}})

    def test_is_sequential(self):
        """
        Test the is_sequential method.
        """
        analyzer = JobDependencyAnalyzer("/some/path")

        job1 = {'start_time': 1000, 'end_time': 1950}
        job2 = {'start_time': 1900, 'end_time': 3000}

        # Validate output for threshold 0.1
        self.assertTrue(analyzer.is_sequential(job1, job2, threshold=0.1))

    def test_is_parallel(self):
        """
        Test the is_parallel method.
        """
        analyzer = JobDependencyAnalyzer("/some/path")

        job1 = {'start_time': 1000, 'end_time': 2000}
        job2 = {'start_time': 1100, 'end_time': 3000}

        # Validate output for threshold 0.1
        self.assertTrue(analyzer.is_parallel(job1, job2, threshold=0.1))

    def test_is_delay(self):
        """
        Test the is_delay method.
        """
        analyzer = JobDependencyAnalyzer("/some/path")

        job1 = {'start_time': 1000, 'end_time': 2000}
        job2 = {'start_time': 2200, 'end_time': 3000}

        # Validate output for threshold 0.1
        self.assertTrue(analyzer.is_delay(job1, job2, threshold=0.1))


if __name__ == "__main__":
    unittest.main()
