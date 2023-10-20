import unittest
import tempfile
import shutil
import os
import json
from workflow_optimizer.workflow_decomposer import WorkflowDecomposer

class TestWorkflowDecomposer(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.wf_folder = os.path.join(self.test_dir, 'sample_workflow')
        os.makedirs(self.wf_folder)

        self.job_folder = os.path.join(self.wf_folder, '123')
        os.makedirs(self.job_folder)

        self.sample_data = {
            "volume": {"timestamp": [1, 2], "bytesRead": [100, 200]},
            "operationsCount": {"timestamp": [1, 2], "operationRead": [1, 2]},
        }

        for file_name, data in self.sample_data.items():
            with open(os.path.join(self.job_folder, f"{file_name}.json"), 'w') as f:
                json.dump(data, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # def test_get_job_timeseries_from_json_file(self):
    #     wf_decomposer = WorkflowDecomposer(workflow_name='sample_workflow')
    #     wf_decomposer.wf_folder = self.wf_folder
    #     result = wf_decomposer.get_job_timeseries_from_json_file('123')

    #     # Here I expect that column_names have the exact number of elements as in DataFrame
    #     # And all columns are accounted for in column_name_map
    #     self.assertIn("volume", result)
    #     self.assertIn("operationsCount", result)
    #     self.assertEqual(result["volume"]["timestamp"], [1, 2])
    #     self.assertEqual(result["volume"]["bytesRead"], [100, 200])
    #     self.assertEqual(result["operationsCount"]["timestamp"], [1, 2])
    #     self.assertEqual(result["operationsCount"]["operationRead"], [1, 2])  # if this is the new name

    def test_process_job_folder(self):
        wf_decomposer = WorkflowDecomposer(workflow_name='sample_workflow')
        wf_decomposer.wf_folder = self.wf_folder
        result = wf_decomposer.process_job_folder('123')
        self.assertIn("123", result)
        self.assertIn("timestamp", result["123"].columns)
        self.assertIn("bytesRead", result["123"].columns)
        self.assertIn("operations_count_read", result["123"].columns)  # Assuming this is the new name after renaming

    # def test_decompose_ioi_job(self):
    #     wf_decomposer = WorkflowDecomposer(workflow_name='sample_workflow')
    #     wf_decomposer.wf_folder = self.wf_folder
    #     print(wf_decomposer.decompose_ioi_job('123'))
    #     representation, phase_features = wf_decomposer.decompose_ioi_job('123')

    #     # Based on your actual implementation, check the representation and phase_features
    #     self.assertIsNotNone(representation)
    #     self.assertIsNotNone(phase_features)

    def test_save_representation(self):
        wf_decomposer = WorkflowDecomposer(workflow_name='sample_workflow')
        wf_decomposer.wf_folder = self.wf_folder
        representation = {"key": "value"}  # Sample representation
        wf_decomposer.save_representation('123', representation)

        # Verify the saved file
        file_path = os.path.join(self.wf_folder, 'decomposed_data', '123_representation.json')
        self.assertTrue(os.path.exists(file_path))

    def test_save_phase_features(self):
        wf_decomposer = WorkflowDecomposer(workflow_name='sample_workflow')
        wf_decomposer.wf_folder = self.wf_folder
        phase_features = {"key": "value"}  # Sample phase features
        wf_decomposer.save_phase_features('123', phase_features)

        # Verify the saved file
        file_path = os.path.join(self.wf_folder, 'decomposed_data', '123_phase_features.json')
        self.assertTrue(os.path.exists(file_path))

    # def test_decompose_workflow(self):
    #     wf_decomposer = WorkflowDecomposer(workflow_name='sample_workflow')
    #     wf_decomposer.wf_folder = self.wf_folder
    #     result = wf_decomposer.decompose_workflow()

    #     # Based on your actual implementation, check the result
    #     self.assertIn("123", result)
    #     self.assertIsNotNone(result["123"]["representation"])
    #     self.assertIsNotNone(result["123"]["phase_features"])

if __name__ == '__main__':
    unittest.main()
