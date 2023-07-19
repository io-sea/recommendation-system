"""This module performs the integration tests for the API Connectors, by calling the endpoints
of the API deployed through the integration platform"""

import os
import unittest
from shutil import rmtree
import urllib3
from pydantic import BaseModel
from typing import List
from app_decomposer.connector import Connector, APIConnector, ApiModel
from app_decomposer.central_job import WorkflowSearcher
from app_decomposer.config_parser import Configuration
from tests_integration.iopa_integration_config import (get_backend_hostname, 
                                                       get_backend_ipv4_port,
                                                       get_keycloak_token, get_mongo_hostname, get_mongo_ipv4_port, 
                                                       MongoDB, __MONGO_DUMP__)

PROTOCOL = "http"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
KIWI_CONFIG = os.path.join(CURRENT_DIR, "test_data", "test_kiwi_config.yaml")
DOCKER_CONFIG = os.path.join(CURRENT_DIR, "test_data", "test_docker_config.yaml")


class TSModel(ApiModel):
    timestamp: int
    bytesRead: int
    bytesWritten: int

class TSListModel(ApiModel):
    timestamp: List[int]
    bytesRead: List[int]
    bytesWritten: List[int]
    
# class TestConnector(unittest.TestCase):
#     def setUp(self):
#         self.keycloak_token = get_keycloak_token()
#         self.backend_hostname = get_backend_hostname()
#         self.backend_ipv4_port = get_backend_ipv4_port()
#         self.connector = Connector(base_url=f"{PROTOCOL}://{self.backend_hostname}:{self.backend_ipv4_port}")  

#     def test_get(self):
#         data = self.connector.get("tools/version")  # Replacez "endpoint" par un endpoint réel
#         print(data)
#         self.assertIsNotNone(data, "La réponse de l'endpoint GET ne devrait pas être None")
        
# class TestKiwiConnector(unittest.TestCase):
#     def setUp(self):
#         """ Prepare the test suite with self.config.
#         Enter path=KIWI_CONFIG for scraping data from kiwi0 IOI slogin node."""
#         self.config = Configuration(path=KIWI_CONFIG)
#         # To disable useless warnings with requests module on https without certificate
#         urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
#     def test_get_token(self):
#         """Test that keycloak_connector.get_kc_token returns a valid token."""
#         keycloak_token = self.config.get_kc_token_v5()
#         print(keycloak_token)
    
class TestDockerConnector(unittest.TestCase):
    def setUp(self):
        """ Prepare the test suite with self.config.
        Enter path=KIWI_CONFIG for scraping data from kiwi0 IOI slogin node."""
        
        self.config = Configuration(path=DOCKER_CONFIG)
        # To disable useless warnings with requests module on https without certificate
        
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
    def test_get_token(self):
        """Test that keycloak_connector.get_kc_token returns a valid token."""
        keycloak_token = self.config.get_kc_token_docker()        
        self.assertIsInstance(keycloak_token, str, "The token should be a string")  
        
    def test_check_connection_iopa_api(self):
        """Tests that checking the connection behaves as expected.
        """
        token = self.config.get_kc_token_docker()
        api_uri = self.config.get_api_uri()
        api_port = self.config.get_api_port()
        
        good_connector = APIConnector(api_uri="http://localhost/pybackend/iopa",
                                      api_token=f"Bearer {token}")
        
        self.assertTrue(good_connector.check_connection())  
         
    def test_check_connection_ioi_api(self):
        """Tests that checking the connection behaves as expected.
        """
        token = self.config.get_kc_token_docker()
        api_uri = self.config.get_api_uri()
        api_port = self.config.get_api_port()
        
        good_connector = APIConnector(api_uri="http://localhost/pybackend/ioi",
                                      api_token=f"Bearer {token}")
        
        self.assertTrue(good_connector.check_connection()) 
        
    def test_api_endpoint(self):
        # Initialize an APIConnector instance with the necessary parameters
        token = self.config.get_kc_token_docker()
        connector = APIConnector(api_uri='http://localhost/pybackend/',
                                api_token=f"Bearer {token}",
                                verify_ssl=False)

        # Set the endpoint and parameters for the request
        endpoint = "/ioi/series/workflow/643776f0547fb888bad22e64"
        params = {"metrics_group": "volume"}

        # Use the request_delegator method to make the GET request
        response = connector.request_delegator("GET", endpoint, params=params)

        data = response.json()
        converted_data = {
            "bytesRead": [item["bytesRead"] for item in data],
            "bytesWritten": [item["bytesWritten"] for item in data],
            "timestamp": [item["timestamp"] for item in data],
            }
        
        # Check the keys in the dictionary
        assert set(converted_data.keys()) == {"bytesRead", "bytesWritten", "timestamp"}

        # Check that the values are lists
        assert isinstance(converted_data["bytesRead"], list)
        assert isinstance(converted_data["bytesWritten"], list)
        assert isinstance(converted_data["timestamp"], list)

    def test_api_workflow_endpoint(self):
        token = self.config.get_kc_token_docker()
        connector = APIConnector(api_uri='http://localhost/pybackend/',
                                api_token=f"Bearer {token}",
                                verify_ssl=False)

        # Set the endpoint for the request
        endpoint = "/ioi/workflows/"

        # Set the parameters for the request
        params = {
            "filtering": [
                {
                    "field": "name",
                    "comparator": "equals",
                    "comparison_value": "Cryo_EM"
                }
            ],
            "order": "asc",
            "sorting_field": "startTime",
            "limit": 50,
            "offset": 0
        }

        # Use the request_delegator method to make the POST request
        response = connector.request_delegator("POST", endpoint, input_json=params)
        # Extract the list of workflow IDs
        workflow_ids = [item['id'] for item in response.json()['data']]
        expected_list = ['642b5fd8547fb888bac488ed',                         
                        '633ab5ebac213a8b8529d186',
                        '634019e6ac213a8b852f5f9c',
                        '63402075ac213a8b852f6821',
                        '634015a2ac213a8b852f4ec5',
                        '63e52bfdac213a8b85e98c21',
                        '63be91efac213a8b85bc56eb',
                        '63be952fac213a8b85bc65c8',
                        '63455953ac213a8b853535f4',
                        '63bfc87bac213a8b85be494b',
                        '643776f0547fb888bad22e64',
                        '634d3df5ac213a8b853d659e',
                        '633313fcac213a8b852232db']
        self.assertListEqual(workflow_ids, expected_list)

class TestWorkflowSearcher(unittest.TestCase):
    def setUp(self):
        """ Prepare the test suite with self.config.
        Enter path=DOCKER_CONFIG for scraping data from kiwi0 IOI slogin node."""

        self.config = Configuration(path=DOCKER_CONFIG)
        token = self.config.get_kc_token_docker()
        self.connector = APIConnector(api_uri='http://localhost/pybackend/',
                                      api_token=f"Bearer {token}",
                                      verify_ssl=False)
        self.searcher = WorkflowSearcher(self.connector)

        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
    def test_search_workflows(self):
        df = self.searcher.search_workflows('Cryo_EM')
        #print(df)
        expected_list = ['642b5fd8547fb888bac488ed',                         
                        '633ab5ebac213a8b8529d186',
                        '634019e6ac213a8b852f5f9c',
                        '63402075ac213a8b852f6821',
                        '634015a2ac213a8b852f4ec5',
                        '63e52bfdac213a8b85e98c21',
                        '63be91efac213a8b85bc56eb',
                        '63be952fac213a8b85bc65c8',
                        '63455953ac213a8b853535f4',
                        '63bfc87bac213a8b85be494b',
                        '643776f0547fb888bad22e64',
                        '634d3df5ac213a8b853d659e',
                        '633313fcac213a8b852232db']
        self.assertListEqual(list(df['id']), expected_list)
        
    def test_extract_workflow_data(self):
        workflow_id = '643776f0547fb888bad22e64'
        data = self.searcher.extract_workflow_data(workflow_id)

        # Check that the result is a dict with the correct keys
        self.assertEqual(set(data.keys()), {workflow_id})

        # Check that the values are dicts with the correct keys
        self.assertEqual(set(data[workflow_id].keys()), {"bytesRead", "bytesWritten", "timestamp"})

        # Check that the values are numpy arrays
        self.assertIsInstance(data[workflow_id]["bytesRead"], np.ndarray)
        self.assertIsInstance(data[workflow_id]["bytesWritten"], np.ndarray)
        self.assertIsInstance(data[workflow_id]["timestamp"], np.ndarray)




        
        
# class TestConnectors(unittest.TestCase):
#     """Parent class for the test of the connectors using the integration platform, which provides:
#         - A class setup dumping the current content into the database in order to restore it
#             once the test series has ended
#         - A class teardown which restores the saved state of the database performed before
#             running the tests.
#         - A setup initializing the logger, getting the information regarding the backend and
#             logging into keycloak
#     """
#     @classmethod
#     def setUpClass(cls):
#         """Initializes a mongoDB connection and save the Experiment content in a JSON file."""
#         cls.db = MongoDB(host=get_mongo_hostname(),
#                          port=get_mongo_ipv4_port())
#         cls.db.dump(__MONGO_DUMP__, collections=['Experiment'])

#     @classmethod
#     def tearDownClass(cls):
#         """Finally restores the Experiment content and clean the dump folder."""
#         cls.db.restore(__MONGO_DUMP__, drop=True)
#         rmtree(__MONGO_DUMP__, ignore_errors=True)

#     def setUp(self):
#         """Initialize the logger
#         """
#         urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#         self.keycloak_token = get_keycloak_token()
#         self.backend_hostname = get_backend_hostname()
#         self.backend_ipv4_port = get_backend_ipv4_port()

#     def test_check_connection(self):
#         """Tests that checking the connection behaves as expected.
#         """
#         good_connector = APIConnector(api_uri=f"{PROTOCOL}://{self.backend_hostname}:{self.backend_ipv4_port}",
#                                       api_token=f"Bearer {self.keycloak_token}",
#                                       mock=False)
#         self.assertTrue(good_connector.check_connection())    

if __name__ == "__main__":
    unittest.main()
    
    
# import unittest
# from shutil import rmtree
# import urllib3

# from app_decomposer.connector import APIConnector
# from tests_integration.iopa_integration_config import get_backend_hostname, get_backend_ipv4_port, \
#     get_keycloak_token, get_mongo_hostname, get_mongo_ipv4_port, MongoDB, __MONGO_DUMP__


# PROTOCOL = "http"


# class TestConnectors(unittest.TestCase):
#     """Parent class for the test of the connectors using the integration platform, which provides:
#         - A class setup dumping the current content into the database in order to restore it
#             once the test series has ended
#         - A class teardown which restores the saved state of the database performed before
#             running the tests.
#         - A setup initializing the logger, getting the information regarding the backend and
#             logging into keycloak
#     """
#     @classmethod
#     def setUpClass(cls):
#         """Initializes a mongoDB connection and save the Experiment content in a JSON file."""
#         cls.db = MongoDB(host=get_mongo_hostname(),
#                          port=get_mongo_ipv4_port())
#         cls.db.dump(__MONGO_DUMP__, collections=['Experiment'])

#     @classmethod
#     def tearDownClass(cls):
#         """Finally restores the Experiment content and clean the dump folder."""
#         cls.db.restore(__MONGO_DUMP__, drop=True)
#         rmtree(__MONGO_DUMP__, ignore_errors=True)

#     def setUp(self):
#         """Initialize the logger
#         """
#         urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#         self.keycloak_token = get_keycloak_token()
#         self.backend_hostname = get_backend_hostname()
#         self.backend_ipv4_port = get_backend_ipv4_port()

#     def test_check_connection(self):
#         """Tests that checking the connection behaves as expected.
#         """
#         good_connector = APIConnector(api_uri=f"{PROTOCOL}://{self.backend_hostname}:{self.backend_ipv4_port}",
#                                       api_token=f"Bearer {self.keycloak_token}",
#                                       mock=False)
#         self.assertTrue(good_connector.check_connection())
