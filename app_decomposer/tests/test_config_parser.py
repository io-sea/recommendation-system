#!/usr/bin/env python
"""
This module proposestests for the configuration parser
"""
from __future__ import division, absolute_import, generators, print_function, unicode_literals, \
    with_statement, nested_scopes  # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2020 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import unittest, os
from app_decomposer.config_parser import Configuration

class TestConfiguration(unittest.TestCase):
    """TestCase to test the functions of the module."""

    def setUp(self):
        """Prepare the test suite."""
        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
        self.yaml_file = os.path.join(CURRENT_DIR, "test_config.yaml")

    def test_configuration(self):
        """Test the 'request_delegator' standard behavior."""
        config = Configuration(path=self.yaml_file).parse()
        self.assertEqual(config["api"]["uri"], 'https://localhost')




if __name__ == '__main__':
    unittest.main(verbosity=2)
