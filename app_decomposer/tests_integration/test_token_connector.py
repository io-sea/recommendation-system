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
from app_decomposer.api_connector import TimeSeries


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_CONFIG_DEFAULT = os.path.join(CURRENT_DIR, "test_data", "test_config.yaml")
TEST_CONFIG_KO = os.path.join(CURRENT_DIR, "test_data", "test_config_kc_ko.yaml")
KIWI_CONFIG = os.path.join(CURRENT_DIR, "test_data", "test_kiwi_config.yaml")

TEST_CONFIG = KIWI_CONFIG
class TestKeycloakToken(unittest.TestCase):
    """ Test KeycloakToken functionalities """

    def setUp(self):
        """ Prepare the test suite with self.config.
        Enter path=KIWI_CONFIG for scraping data from kiwi0 IOI slogin node."""
        self.config = Configuration(path=TEST_CONFIG)
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def test_get_token(self):
        """Test that keycloak_connector.get_kc_token returns a valid token."""
        keycloak_token = self.config.get_kc_token()
        #self.assertIn('Bearer ', keycloak_token)
        # test to call an api with the token
        api_uri = f"{self.config.get_api_uri()}:{self.config.get_api_port()}" \
            "/backend/api/user/settings"

        rqst = request_delegator(requests.get,
                                 api_uri,
                                 headers={'Authorization': "Bearer "+keycloak_token})
        self.assertIsInstance(rqst, requests.Response)
        resp = rqst.json()
        self.assertIn('username', resp)
        self.assertEqual(resp['username'], "ioi-admin")

    @unittest.skipUnless(TEST_CONFIG == KIWI_CONFIG, "Skip this test because kiwi0 is not up")
    def test_on_kiwi0(self):
        """test if unittest activates on condition"""
        self.assertEqual(TEST_CONFIG, KIWI_CONFIG)

    @unittest.skipUnless(TEST_CONFIG == KIWI_CONFIG, "Skip this test because kiwi0 is not up")
    def test_get_ts_on_kiwi0(self):
        """tests if can get timeseries from kiwi0.
        Details on output format
        {
            'volume': {
                'timestamp': array([1666359470000, 1666359475000, 1666359480000], dtype=int64), 'bytesRead': array([10153284,  9848000,        0], dtype=int64),
                'bytesWritten': array([ 0,  0, 42], dtype=int64)},
            'operationsCount': {
                'timestamp': array([1666359470000, 1666359475000, 1666359480000], dtype=int64), 'operationRead': array([10160,  9850,     1], dtype=int64),
                'operationWrite': array([0, 0, 1], dtype=int64)},
            'accessPattern': {
                'timestamp': array([1666359470000, 1666359475000, 1666359480000], dtype=int64), 'accessRandRead': array([0, 0, 0], dtype=int64),
                'accessSeqRead': array([10120,  9800,     0], dtype=int64),
                'accessStrRead': array([0, 0, 0], dtype=int64),
                'accessUnclRead': array([40, 50,  1], dtype=int64),
                'accessRandWrite': array([0, 0, 0], dtype=int64),
                'accessSeqWrite': array([0, 0, 0], dtype=int64),
                'accessStrWrite': array([0, 0, 0], dtype=int64),
                'accessUnclWrite': array([0, 0, 1], dtype=int64)
                }
            }
        """

        token = self.config.get_kc_token()
        # To disable useless warnings with requests module on https without certificate
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        api_uri = f"{self.config.get_api_uri()}:{self.config.get_api_port()}"
        ts_data = dict()
        for ts in ["volume", "operationsCount", "accessPattern"]:
            ts_data[ts]  = TimeSeries(api_uri=api_uri,
                                      api_token=f"Bearer {token}",
                                      object_id='6352a0af7992958a4b807942',
                                      type_series=ts).get_data_by_label()





if __name__ == '__main__':
    unittest.main(verbosity=2)
