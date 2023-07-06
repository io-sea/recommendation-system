"""This module performs the integration tests for the API Connectors, by calling the endpoints
of the API deployed through the integration platform"""

import os
import unittest
from shutil import rmtree
import urllib3
from app_decomposer.connector import Connector, APIConnector, ApiModel  
from app_decomposer.config_parser import Configuration
from tests_integration.iopa_integration_config import (get_backend_hostname, 
                                                       get_backend_ipv4_port,
                                                       get_keycloak_token, get_mongo_hostname, get_mongo_ipv4_port, 
                                                       MongoDB, __MONGO_DUMP__)

PROTOCOL = "http"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
KIWI_CONFIG = os.path.join(CURRENT_DIR, "test_data", "test_kiwi_config.yaml")

class TestConnector(unittest.TestCase):
    def setUp(self):
        self.keycloak_token = get_keycloak_token()
        self.backend_hostname = get_backend_hostname()
        self.backend_ipv4_port = get_backend_ipv4_port()
        self.connector = Connector(base_url=f"{PROTOCOL}://{self.backend_hostname}:{self.backend_ipv4_port}")  

    def test_get(self):
        data = self.connector.get("tools/version")  # Replacez "endpoint" par un endpoint réel
        print(data)
        self.assertIsNotNone(data, "La réponse de l'endpoint GET ne devrait pas être None")
        
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
    
        
        
class TestConnectors(unittest.TestCase):
    """Parent class for the test of the connectors using the integration platform, which provides:
        - A class setup dumping the current content into the database in order to restore it
            once the test series has ended
        - A class teardown which restores the saved state of the database performed before
            running the tests.
        - A setup initializing the logger, getting the information regarding the backend and
            logging into keycloak
    """
    @classmethod
    def setUpClass(cls):
        """Initializes a mongoDB connection and save the Experiment content in a JSON file."""
        cls.db = MongoDB(host=get_mongo_hostname(),
                         port=get_mongo_ipv4_port())
        cls.db.dump(__MONGO_DUMP__, collections=['Experiment'])

    @classmethod
    def tearDownClass(cls):
        """Finally restores the Experiment content and clean the dump folder."""
        cls.db.restore(__MONGO_DUMP__, drop=True)
        rmtree(__MONGO_DUMP__, ignore_errors=True)

    def setUp(self):
        """Initialize the logger
        """
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.keycloak_token = get_keycloak_token()
        self.backend_hostname = get_backend_hostname()
        self.backend_ipv4_port = get_backend_ipv4_port()

    def test_check_connection(self):
        """Tests that checking the connection behaves as expected.
        """
        good_connector = APIConnector(api_uri=f"{PROTOCOL}://{self.backend_hostname}:{self.backend_ipv4_port}",
                                      api_token=f"Bearer {self.keycloak_token}",
                                      mock=False)
        self.assertTrue(good_connector.check_connection())    

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
