import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open
from workflow_optimizer.job_dependency_analyzer_nx import JobDependencyAnalyzerNX

class TestJobDependencyAnalyzerNX(unittest.TestCase):
    """
    Unit tests for the JobDependencyAnalyzer class.
    """


    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.path.exists")
    def test_extract_and_sort_jobs_no_volume_json(self, mock_exists, mock_isdir, mock_listdir):
        """
        Test the extract_and_sort_jobs method when there is no volume.json file.
        """

        # Mocking behavior
        mock_listdir.return_value = ['job1', 'job2']
        mock_isdir.side_effect = lambda x: True
        mock_exists.side_effect = lambda x: False if 'volume.json' in x else True

        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        sorted_jobs = analyzer.extract_and_sort_jobs()

        expected = {}  # No jobs should be returned

        self.assertEqual(sorted_jobs, expected)

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

        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        sorted_jobs = analyzer.extract_and_sort_jobs()

        expected = {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 1000, 'end_time': 2000}
        }

        self.assertEqual(sorted_jobs, expected)

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_sequential_true(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.threshold = 0.1
        # Mock job data
        job1 = {'start_time': 100, 'end_time': 200}
        job2 = {'start_time': 201, 'end_time': 300}

        # Expect is_sequential to be True
        self.assertTrue(analyzer.is_sequential(job1, job2))

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_sequential_false(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.threshold = 0.1
        # Mock job data
        job1 = {'start_time': 100, 'end_time': 200}
        job2 = {'start_time': 300, 'end_time': 400}

        # Expect is_sequential to be False since job2 starts much later than job1 ends
        self.assertFalse(analyzer.is_sequential(job1, job2))

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_sequential_zero_threshold(self, MockInit):

        MockInit.return_value = None
        # Set up the analyzer with a zero threshold
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.', threshold=0)

        # Define jobs that end and start exactly at the same time
        job1 = {'start_time': 100, 'end_time': 200}
        job2 = {'start_time': 200, 'end_time': 300}

        # With a zero threshold, this should be sequential
        self.assertTrue(analyzer.is_sequential(job1, job2))

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_parallel_true(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.threshold = 0.1
        # Mock job data where job1 and job2 overlap in time
        job1 = {'start_time': 100, 'end_time': 200}
        job2 = {'start_time': 150, 'end_time': 250}

        # Expect is_parallel to be True
        self.assertTrue(analyzer.is_parallel(job1, job2))

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_parallel_false(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.threshold = 0.1
        # Mock job data where job1 ends before job2 starts
        job1 = {'start_time': 100, 'end_time': 150}
        job2 = {'start_time': 200, 'end_time': 250}

        # Expect is_parallel to be False
        self.assertFalse(analyzer.is_parallel(job1, job2))

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_is_parallel_exact_overlap(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None
        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.threshold = 0
        # Define jobs that start and end at the same time
        job1 = {'start_time': 100, 'end_time': 200}
        job2 = {'start_time': 100, 'end_time': 200}

        # These jobs are running in parallel
        self.assertTrue(analyzer.is_parallel(job1, job2))


    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_analyze_dependencies_no_jobs(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize the analyzer with an empty job list
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.sorted_jobs = {}

        # Analyze dependencies should handle empty job lists gracefully
        analyzer.analyze_dependencies()
        # The graph should be empty
        self.assertEqual(len(analyzer.graph.edges()), 0)


    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_analyze_dependencies(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.threshold = 0.1

        # Set sorted_jobs directly

        analyzer.sorted_jobs = {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 2100, 'end_time': 2500},
            'job3': {'start_time': 2200, 'end_time': 3000}
        }

        analyzer.analyze_dependencies()
        dependencies = analyzer.graph
        for edge in analyzer.graph.edges(data=True):
            print(edge)
        self.assertTrue(analyzer.graph.has_edge('job1', 'job2'))
        self.assertEqual(analyzer.graph['job1']['job2']['type'], 'sequential')
        self.assertTrue(analyzer.graph.has_edge('job1', 'job3'))
        self.assertEqual(analyzer.graph['job1']['job3']['type'], 'parallel')
        self.assertTrue(analyzer.graph.has_edge('job2', 'job3'))
        self.assertEqual(analyzer.graph['job2']['job3']['type'], 'parallel')


    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_find_nodes_with_no_multiple_incoming_edges(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.graph.add_edges_from([
            ('job1', 'job2', {'type': 'sequential'}),
            ('job2', 'job3', {'type': 'sequential'})
        ])

        # Expect an empty list
        self.assertEqual(analyzer.find_nodes_with_multiple_incoming_edges(), [])


    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_remove_edges_by_priority_same_priority(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.graph.add_edges_from([
            ('job1', 'job3', {'type': 'parallel'}),
            ('job2', 'job3', {'type': 'parallel'})
        ])

        # Run the method to remove edges by priority
        analyzer.remove_edges_by_priority('job3')

        # One edge should remain, ensure the correct logic is applied to decide which one
        self.assertEqual(len(analyzer.graph.in_edges('job3')), 1)

    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_clean_redundancy_no_redundancies(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.graph.add_edges_from([
            ('job1', 'job2', {'type': 'sequential'}),
            ('job2', 'job3', {'type': 'sequential'})
        ])

        # Run the redundancy cleaning method
        analyzer.clean_redundancy()

        # No changes should be made to the graph
        self.assertEqual(len(analyzer.graph.edges()), 2)


    @patch("workflow_optimizer.job_dependency_analyzer.JobDependencyAnalyzer.__init__")
    def test_find_redundant_nodes(self, MockInit):
        """
        Test the analyze_dependencies method.
        """
        # Mock the __init__ so it doesn't do anything
        MockInit.return_value = None

        # Initialize mock object
        analyzer = JobDependencyAnalyzerNX(workflow_folder='.')
        analyzer.threshold = 0.1

        # Set sorted_jobs directly

        analyzer.sorted_jobs = {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 2100, 'end_time': 2500},
            'job3': {'start_time': 2200, 'end_time': 3000}
        }

        analyzer.analyze_dependencies()

        self.assertEqual(analyzer.graph['job1']['job2']['type'], 'sequential')
        self.assertEqual(analyzer.graph['job2']['job3']['type'], 'delay')
        self.assertEqual(analyzer.graph['job2']['job3']['delay'], 100)


class TestJobDependencyAnalyzerNXFunctional(unittest.TestCase):
    """
    Functional tests for the JobDependencyAnalyzerNX class.
    """

    def setUp(self):
        """
        Set up a temporary directory with mock job data for testing.
        """
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create mock data for jobs
        self.jobs_data = {
            'job1': {'start_time': 1000, 'end_time': 2000},
            'job2': {'start_time': 1050, 'end_time': 3000},
            'job3': {'start_time': 3000, 'end_time': 5300}
        }

        # Populate the temporary directory with job folders and volume.json files
        for job, times in self.jobs_data.items():
            job_dir = os.path.join(self.test_dir, job)
            os.makedirs(job_dir)
            volume_data = [[times['start_time'], 0], [times['end_time'], 0]]
            with open(os.path.join(job_dir, 'volume.json'), 'w') as f:
                json.dump(volume_data, f)

    def tearDown(self):
        """
        Clean up by removing the temporary directory after tests.
        """
        shutil.rmtree(self.test_dir)

    def test_functional_workflow_analysis(self):
        """
        Test the JobDependencyAnalyzerNX with a functional workflow analysis.
        """
        # Create an instance of the analyzer with the temporary directory
        analyzer = JobDependencyAnalyzerNX(workflow_folder=self.test_dir)

        # Run the analysis
        analyzer.analyze_dependencies()

        # Check the graph has the correct edges with the correct types
        expected_edges = [
            ('job1', 'job2', "parallel"),
            ('job2', 'job3', "sequential")
        ]

        actual_edges = [(u, v, d['type']) for u, v, d in analyzer.graph.edges(data=True)]
        self.assertEqual(sorted(actual_edges), sorted(expected_edges))

        self.assertEqual(analyzer.graph['job1']['job2']['type'], 'parallel')
        self.assertEqual(analyzer.graph['job2']['job3']['type'], 'sequential')

if __name__ == "__main__":
    unittest.main()
