import unittest
from workflow_optimizer.workflow_synthesizer import WorkflowSynthesizer
import pandas as pd
import json

class TestWorkflowSynthesizer(unittest.TestCase):
    """
    This class contains unit tests for the WorkflowSynthesizer class.
    """

    def setUp(self):
        """
        This method sets up the test data and initializes the WorkflowSynthesizer object.
        """
        self.synthesizer = WorkflowSynthesizer()

        # Timestamps updated to respect a minimum delta of 5000
        data_for_jobs = {
            'job1': pd.DataFrame({'timestamp': [5000, 10000, 15000],
                                  'bytesRead': [10, 20, 30],
                                  'bytesWritten': [5, 15, 25]}),
            'job2': pd.DataFrame({'timestamp': [10000, 20000, 30000],
                                  'bytesRead': [40, 50, 60],
                                  'bytesWritten': [35, 45, 55]})
        }

        self.synthesizer.synthesize(data_for_jobs)

    def test_to_dict(self):
        """
        This method tests the to_dict() method of the WorkflowSynthesizer class.
        """
        expected_output = {
            'timestamp': [5000, 10000, 15000, 20000, 25000, 30000],
            'bytesRead': [10, 60, 30, 50, 0, 60],
            'bytesWritten': [5, 50, 25, 45, 0, 55]
        }

        self.assertEqual(self.synthesizer.to_dict(), expected_output)

    def test_export_to_json(self):
        """
        This method tests the export_to_json() method of the WorkflowSynthesizer class.
        """
        self.synthesizer.export_to_json('test_output.json')

        with open('test_output.json', 'r') as f:
            data = json.load(f)

        expected_output = {
            'timestamp': [100, 150, 200, 250, 300, 350],
            'bytesRead': [10, 40, 20, 50, 30, 60],
            'bytesWritten': [5, 35, 15, 45, 25, 55]
        }

        self.assertEqual(data, expected_output)


if __name__ == '__main__':
    unittest.main()
