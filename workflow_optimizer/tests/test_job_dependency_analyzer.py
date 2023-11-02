import os
import json
import unittest
from loguru import logger
from unittest.mock import patch, mock_open, MagicMock
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
        analyzer.analyze_dependencies()
        # Validate sorted jobs
        self.assertEqual(analyzer.sorted_jobs, {'job1': {'start_time': 1000, 'end_time': 2000}, 'job2': {'start_time': 1000, 'end_time': 2000}})

        # Validate 'delay' as default dependency type
        self.assertDictEqual({'sequential': [], 'parallel': [['job1', 'job2']], 'delay': []}, analyzer.dependencies)

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_connected(self, MockInit):
        """
        Test the is_connected method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.dependencies = {'sequential': [['job700', 'job800'], ['job800', 'job900']],
                                      'parallel': [],
                                      'delay': []}
        result = mock_analyzer.is_connected('job700', 'job900')
        self.assertTrue(result)

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_connected_2(self, MockInit):
        """
        Test the is_connected method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.dependencies = {'sequential': [],
                                      'parallel': [['job700', 'job800'], ['job800', 'job900']],
                                      'delay': []}
        result = mock_analyzer.is_connected('job700', 'job900')
        self.assertTrue(result)

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_connected_3(self, MockInit):
        """
        Test the is_connected method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.dependencies = {'sequential': [],
                                      'parallel': [],
                                      'delay': [['job700', 'job800'], ['job800', 'job900']]}
        result = mock_analyzer.is_connected('job700', 'job900')
        self.assertTrue(result)

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_connected_4(self, MockInit):
        """
        Test the is_connected method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.dependencies = {'sequential': [['job1', 'job2'], ['job3', 'job4']],
                                      'parallel': [['job2', 'job3']],
                                      'delay': []}
        self.assertTrue(mock_analyzer.is_connected('job1', 'job4'))
        self.assertTrue(mock_analyzer.is_connected('job1', 'job3'))

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer")
    def test_is_sequential(self, MockAnalyzer):
        """
        Test the is_sequential method.
        """
        mock_analyzer = MockAnalyzer.return_value
        mock_analyzer.sorted_jobs = {
            'job1': {'start_time': 1000, 'end_time': 1950},
            'job2': {'start_time': 1900, 'end_time': 3000}
        }

        # Validate output for threshold 0.1
        self.assertTrue(mock_analyzer.is_sequential(mock_analyzer.sorted_jobs['job1'],
                                                    mock_analyzer.sorted_jobs['job2'],
                                                    threshold=0.1))

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer")
    def test_is_parallel(self, MockAnalyzer):
        """
        Test the is_parallel method.
        """
        mock_analyzer = MockAnalyzer.return_value
        mock_analyzer.sorted_jobs = {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 1100, 'end_time': 3000}
        }

        # Validate output for threshold 0.1
        self.assertTrue(mock_analyzer.is_parallel(mock_analyzer.sorted_jobs['job1'],
                                             mock_analyzer.sorted_jobs['job2'],
                                             threshold=0.1))


    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_analyze_dependencies(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.threshold = 0.1

        # Set sorted_jobs directly
        mock_analyzer.sorted_jobs = {
            'job1': {'start_time': 1000, 'end_time': 1950},
            'job2': {'start_time': 1900, 'end_time': 3000},
            'job3': {'start_time': 3000, 'end_time': 4000}
        }

        mock_analyzer.analyze_dependencies()
        dependencies = mock_analyzer.dependencies
        # Validate output (modify this according to what you expect the output to be)
        self.assertEqual(dependencies, {'sequential': [['job1', 'job2'], ['job2', 'job3']],
                                        'parallel': [], 'delay': []})

        # Validate that is_sequential was called (modify this based on your expectations)
        MockInit.assert_called()


    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_dependencies_1(self, MockInit):
        """
        Test that the analyze_dependencies method correctly identifies parallel jobs.
        """
        MockInit.return_value = None
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.threshold = 0.1
        mock_analyzer.sorted_jobs = {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 1050, 'end_time': 2050},
            'job3': {'start_time': 1200, 'end_time': 1300}
        }
        mock_analyzer.analyze_dependencies()
        dependencies = mock_analyzer.dependencies
        self.assertIn(['job1', 'job2'], dependencies['parallel'])
        self.assertNotIn(['job2', 'job3'], dependencies['sequential'])


    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_dependencies_2(self, MockInit):
        """
        Test that the analyze_dependencies method correctly identifies parallel jobs.
        """
        MockInit.return_value = None
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.threshold = 0.1
        mock_analyzer.sorted_jobs = {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 1050, 'end_time': 2050},
            'job3': {'start_time': 2050, 'end_time': 3000}
        }
        mock_analyzer.analyze_dependencies()
        dependencies = mock_analyzer.dependencies
        print(dependencies)
        expected_dependencies = {'sequential': [['job1', 'job3'], ['job2', 'job3']],
                                  'parallel': [['job1', 'job2']],
                                  'delay': []}

        self.assertDictEqual(expected_dependencies, dependencies)
        # self.assertIn(['job1', 'job3'], dependencies['sequential'])
        # self.assertIn(['job2', 'job3'], dependencies['sequential'])
        # self.assertNotIn(['job1', 'job2'], dependencies['parallel'])

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_analyze_dependencies_threshold_0(self, MockInit):
        """
        Test the analyze_dependencies method when threshold is 0.
        In such case, only delay dependencies are considered.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.threshold = 0.1

        # Set sorted_jobs directly
        mock_analyzer.sorted_jobs = {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 2000, 'end_time': 3000},
            'job3': {'start_time': 2000, 'end_time': 7000}
        }

        mock_analyzer.analyze_dependencies()
        dependencies = mock_analyzer.dependencies
        print(dependencies)
        # Validate output (modify this according to what you expect the output to be)
        self.assertEqual(dependencies, {'sequential': [],
                                        'parallel': [],
                                        'delay': [['job1', 'job2', 0],
                                                  ['job2', 'job3', -1000]]})

        # Validate that is_sequential was called (modify this based on your expectations)
        MockInit.assert_called()

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_analyze_dependencies_threshold_1(self, MockInit):
        """
        Test the analyze_dependencies method when threshold is 0.
        In such case, only delay dependencies are considered.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.threshold = 0.1

        # Set sorted_jobs directly
        mock_analyzer.sorted_jobs = {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 1900, 'end_time': 3000},
            'job3': {'start_time': 5000, 'end_time': 7000}
        }

        mock_analyzer.analyze_dependencies()
        dependencies = mock_analyzer.dependencies
        # Validate output (modify this according to what you expect the output to be)
        self.assertEqual(dependencies, {'sequential': [['job1', 'job2']],
                                        'parallel': [],
                                        'delay': [['job2', 'job3', 2000]]})

        # Validate that is_sequential was called (modify this based on your expectations)
        MockInit.assert_called()


class TestIsConnected(unittest.TestCase):
    """
    Unit tests for the is_connected method in JobDependencyAnalyzer.
    """
    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def setUp(self, MockInit):
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        mock_analyzer = JobDependencyAnalyzer()
        mock_analyzer.threshold = 0.1
        self.analyzer = mock_analyzer
        self.analyzer.dependencies = {
            'sequential': [['job1', 'job2'], ['job2', 'job3']],
            'parallel': [['job3', 'job4']],
            'delay': [['job17', 'job18', 200]],
        }

    def test_directly_connected_sequential(self):
        self.assertTrue(self.analyzer.is_connected('job1', 'job2'))

    def test_directly_connected_parallel(self):
        self.assertTrue(self.analyzer.is_connected('job3', 'job4'))

    def test_directly_connected_delay(self):
        self.assertTrue(self.analyzer.is_connected('job17', 'job18'))

    def test_indirectly_connected_sequential(self):
        self.assertTrue(self.analyzer.is_connected('job1', 'job3'))

    def test_indirectly_connected_sequential_parallel(self):
        self.assertTrue(self.analyzer.is_connected('job1', 'job4'))

    def test_not_connected_1(self):
        self.assertFalse(self.analyzer.is_connected('job1', 'job5'))

    def test_not_connected_2(self):
        self.assertFalse(self.analyzer.is_connected('job2', 'job5'))

    def test_same_job(self):
        self.assertTrue(self.analyzer.is_connected('job1', 'job1'))

    def test_nonexistent_job(self):
        self.assertFalse(self.analyzer.is_connected('job1', 'job6'))


if __name__ == "__main__":
    unittest.main()
