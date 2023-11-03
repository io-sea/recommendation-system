import unittest
import json
from unittest.mock import patch, mock_open
from workflow_optimizer.job_dependency_analyzer_nx import JobDependencyAnalyzerNX

class TestJobDependencyAnalyzerNX(unittest.TestCase):
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
        # # display before
        # print(f"before cleaning: \n\t{analyzer.graph.edges(data=True)}")
        # #analyzer.remove_redundant_dependencies()
        # redundant_nodes = analyzer.find_nodes_with_multiple_incoming_edges()
        # # display after cleaning
        # #print(f"After cleaning: \n\t{analyzer.graph.edges(data=True)}")
        # # Print the nodes with their incoming edge counts
        # for node, count in redundant_nodes:
        #     print(f"Node '{node}' has {count} incoming edges.")
        #     analyzer.remove_edges_by_priority(node)
        # print(f"After cleaning: \n\t{analyzer.graph.edges(data=True)}")

        analyzer.clean_redundancy()
        print(f"After cleaning: \n\t{analyzer.graph.edges(data=True)}")


if __name__ == "__main__":
    unittest.main()
