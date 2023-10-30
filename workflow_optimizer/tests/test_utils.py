import unittest
from unittest.mock import patch
from workflow_optimizer import utils
from loguru import logger


class TestUtilFunctions(unittest.TestCase):

    def test_camel_case_to_snake_case(self):
        self.assertEqual(utils.camel_case_to_snake_case("CamelCase"), "camel_case")
        self.assertEqual(utils.camel_case_to_snake_case("camelCase"), "camel_case")
        self.assertEqual(utils.camel_case_to_snake_case("Camel"), "camel")
        self.assertEqual(utils.camel_case_to_snake_case("C"), "c")

    def test_get_column_names(self):
        self.assertEqual(utils.get_column_names("volume.json"), ["timestamp", "bytesRead", "bytesWritten"])
        self.assertEqual(utils.get_column_names("operationsCount.json"), ["timestamp", "operations_count_read", "operations_count_write"])
        self.assertEqual(utils.get_column_names("non_existent.json"), [])

    def test_is_file_extension(self):
        self.assertTrue(utils.is_file_extension("example.json", "json"))
        self.assertFalse(utils.is_file_extension("example.txt", "json"))
        self.assertFalse(utils.is_file_extension("example.json", "txt"))
        self.assertFalse(utils.is_file_extension("example", "json"))

    def test_list_and_classify_directory_contents(self):
        # This test may require mocking the file system, or you can create a test directory
        # and then verify the output of utils.list_and_classify_directory_contents().
        # For the purpose of this example, we'll skip this test.
        pass

    class TestListAndClassifyDirectoryContents(unittest.TestCase):

        @patch('your_package_name.utils.os.path.isdir')  # Replace 'your_package_name' with the actual package name
        @patch('your_package_name.utils.os.path.isfile')  # Replace 'your_package_name' with the actual package name
        @patch('your_package_name.utils.os.listdir')  # Replace 'your_package_name' with the actual package name
        def test_list_and_classify(self, mock_listdir, mock_isfile, mock_isdir):
            # Mocking the behavior of os.listdir to return a specific list
            mock_listdir.return_value = ['file1.txt', 'file2.txt', 'folder1', 'folder2', 'unknown']

            # Mocking the behavior of os.path.isfile and os.path.isdir
            mock_isfile.side_effect = lambda x: x in ['file1.txt', 'file2.txt']
            mock_isdir.side_effect = lambda x: x in ['folder1', 'folder2']

            # Capturing the standard output
            with patch('builtins.print') as mock_print:
                utils.list_and_classify_directory_contents('some_path')
                # Replace 'your_package_name' with the actual package name

            # Checking if the print statements are as expected
            mock_print.assert_any_call('file1.txt -> File')
            mock_print.assert_any_call('file2.txt -> File')
            mock_print.assert_any_call('folder1 -> Folder')
            mock_print.assert_any_call('folder2 -> Folder')
            mock_print.assert_any_call('unknown -> Unknown')

if __name__ == '__main__':
    unittest.main()
