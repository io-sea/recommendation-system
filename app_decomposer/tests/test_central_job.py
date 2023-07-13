import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from app_decomposer.central_job import WorkflowSynthesizer, CentralJob, WorkflowSearcher


class TestWorkflowSynthesizer(unittest.TestCase):

    def setUp(self):
        self.ws = WorkflowSynthesizer()

        self.job1 = {
            'bytesRead': np.array([6457, 0, 1090522169, 2055211225]),
            'bytesWritten': np.array([115404800, 191795200, 0, 0]),
            'timestamp': np.array([1581607510, 1581607515, 1581607520, 1581607525])
        }

        self.job2 = {
            'bytesRead': np.array([2457, 8000, 0, 0]),
            'bytesWritten': np.array([15404800, 91795200, 0, 0]),
            'timestamp': np.array([1581607520, 1581607525, 1581607530, 1581607535])
        }

        self.job3 = {
            'bytesRead': np.array([6457, 0, 1090522169, 2055211225]),
            'bytesWritten': np.array([115404800, 191795200, 0, 0]),
            'timestamp': np.array([1581607530, 1581607535, 1581607540, 1581607545])
        }

    def test_synthetize(self):
        self.ws.synthetize([self.job1, self.job2, self.job3])
        self.assertIsInstance(self.ws.workflow, pd.DataFrame)
        self.assertEqual(len(self.ws.workflow), 8)
        self.assertEqual(self.ws.workflow['sumBytesRead'].sum(),
                         self.job1['bytesRead'].sum() +
                         self.job2['bytesRead'].sum() +
                         self.job3['bytesRead'].sum())
        self.assertEqual(self.ws.workflow['sumBytesWritten'].sum(), 
                         self.job1['bytesWritten'].sum() + 
                         self.job2['bytesWritten'].sum() + 
                         self.job3['bytesWritten'].sum())

    def test_to_dict(self):
        self.ws.synthetize([self.job1, self.job2, self.job3])

        result = self.ws.to_dict()

        self.assertIsInstance(result, dict)
        self.assertListEqual(result['timestamp'], 
                             self.ws.workflow['index'].tolist())
        self.assertListEqual(result['bytesRead'], 
                             self.ws.workflow['sumBytesRead'].tolist())
        self.assertListEqual(result['bytesWritten'], 
                             self.ws.workflow['sumBytesWritten'].tolist())



class TestWorkflowSearcher(unittest.TestCase):
    def setUp(self):
        # Mock APIConnector object
        self.mock_connector = MagicMock()
        self.searcher = WorkflowSearcher(self.mock_connector)

    def test_search_workflows(self):
        # Mock the response from the connector
        self.mock_connector.request_delegator.return_value.json.return_value = {
            'data': [
                {'id': '1', 'name': 'Workflow1'},
                {'id': '2', 'name': 'Workflow2'}
            ]
        }

        df = self.searcher.search_workflows('Workflow1')

        # Check that request_delegator was called with the right arguments
        self.mock_connector.request_delegator.assert_called_with(
            "POST",
            "/ioi/workflows/",
            input_json={
                "filtering": [
                    {
                        "field": "name",
                        "comparator": "equals",
                        "comparison_value": 'Workflow1'
                    }
                ],
                "order": "asc",
                "sorting_field": "startTime",
                "limit": 50,
                "offset": 0
            }
        )

        # Check that the result is a DataFrame with the correct columns
        self.assertEqual(list(df.columns), ['id', 'name'])

    def test_extract_workflow_data(self):
        # Mock the response from the connector
        self.mock_connector.request_delegator.return_value.json.return_value = [
            {"bytesRead": 100, "bytesWritten": 200, "timestamp": 1234567890}
        ]

        data = self.searcher.extract_workflow_data('1')

        # Check that request_delegator was called with the right arguments
        self.mock_connector.request_delegator.assert_called_with(
            "GET",
            "/ioi/series/workflow/1",
            params={"metrics_group": "volume"}
        )

        # Check that the result is a dict with the correct keys and values
        self.assertEqual(data, {
            "bytesRead": [100],
            "bytesWritten": [200],
            "timestamp": [1234567890],
        })



class TestCentralJob(unittest.TestCase):
    """
    Class to test the CentralJob class.
    """

    def setUp(self):
        """
        Setup for the tests. We'll use this to create some default jobs.
        """
        self.jobs = {
            "job_1": {
                'bytesRead': np.array([6457, 0, 1090522169, 2055211225]),
                'bytesWritten': np.array([115404800, 191795200, 0, 0])
            },
            "job_2": {
                'bytesRead': np.array([6957, 0, 1070522169, 2025211225]),
                'bytesWritten': np.array([120404800, 198795200, 0, 0])
            }
        }

    def test_init(self):
        """
        Tests that the CentralJob class initializes correctly.
        """
        cj = CentralJob(self.jobs)
        self.assertEqual(cj.jobs, self.jobs)
        self.assertEqual(cj.n_components, 20)
        self.assertEqual(cj.normalization_type, 'minmax')
        self.assertIsNone(cj.features)

    def test_process(self):        
        # Mocking jobs data
        jobs = {
            "job_1": {"bytesRead": np.array([1,2,3,4,5]), "bytesWritten": np.array([2,3,4,5,6])},
            "job_2": {"bytesRead": np.array([2,3,4,5,6]), "bytesWritten": np.array([3,4,5,6,7])}
        }
        # Init the class
        self.cj = CentralJob(jobs=jobs)
        # Process the job data
        features = self.cj.process()
        # Assertions
        self.assertIsInstance(features, list)  # assert features is a list
        self.assertEqual(len(features), len(jobs))  # assert two jobs' features are processed
        for feature_set in features:
            self.assertEqual(len(feature_set), 
                             2*(3+self.cj.n_components))  # assert each feature set contains correct number of elements

    def test_find_central_job(self):
        """
        Test the find_central_job function.
        """
        cj = CentralJob(self.jobs)
        idx = cj.find_central_job()        
        # Check that the returned index is valid
        self.assertTrue(0 <= idx < len(self.jobs))

if __name__ == '__main__':
    unittest.main()