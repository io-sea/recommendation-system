#!/usr/bin/env python
"""
This module proposes integration tests for the token connector module
"""
from __future__ import division, absolute_import, generators, print_function, unicode_literals, \
    with_statement, nested_scopes  # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2020 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import os
import unittest
import requests
import urllib3

from app_decomposer.api_connector import request_delegator
from app_decomposer.config_parser import Configuration


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_CONFIG = os.path.join(CURRENT_DIR, "test_data", "test_config.yaml")
TEST_CONFIG_KO = os.path.join(CURRENT_DIR, "test_data", "test_config_kc_ko.yaml")

class TestKeycloakToken(unittest.TestCase):
    """ Test KeycloakToken functionalities """

    def setUp(self):
        """ Prepare the test suite """
        self.config = Configuration(path=TEST_CONFIG)
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def test_get_token(self):
        """Test that keycloak_connector.get_kc_token returns a valid token."""
        keycloak_token = self.config.get_kc_token()
        print(keycloak_token)
        self.assertIn('Bearer ', keycloak_token)
        # test to call an api with the token
        # api_uri = f"{self.config.get_api_uri()}:{self.config.get_api_port()}" \
        #     "/backend/api/user/settings"
        # rqst = request_delegator(requests.get,
        #                          api_uri,
        #                          headers={'Authorization': keycloak_token})
        # self.assertIsInstance(rqst, requests.Response)
        # resp = rqst.json()
        # self.assertIn('username', resp)
        # self.assertEqual(resp['username'], "ioi-admin")

    # def test_check_connection(self):
    #     """Test that keycloak_connector.check_connection returns True, if the connection to the
    #     server can be reached."""
    #     self.assertTrue(self.keycloak_connector.check_connection())

    # def test_check_connection_ko(self):
    #     """Test that keycloak_connector.check_connection returns False, if the connection to the
    #     server cannot be reached."""
    #     keycloak_config = KeycloakConfig.from_yaml(TEST_CONFIG_KO)
    #     keycloak_connector = KeycloakConnector(keycloak_config.keycloak, 'ioi-admin', 'password')
    #     self.assertFalse(keycloak_connector.check_connection())


if __name__ == '__main__':
    unittest.main(verbosity=2)
