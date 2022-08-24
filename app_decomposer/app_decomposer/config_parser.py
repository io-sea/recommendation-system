"""
This module parses a YAML configuration file.
"""

__copyright__ = """
Copyright (C) 2021 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

from pathlib import Path
import yaml
from app_decomposer import DEFAULT_CONFIGURATION


class Configuration:
    """Base class to load a YAML file and parse its content. """

    def __init__(self, path):
        """Loads the yaml file located at the path 'path'.

        Args:
            path (str): the path of the yaml file.

        Returns:
            A dict with configuration content and parameters.
        """
        self.parsed_yaml_file = yaml.load(Path(path).read_text(), Loader=yaml.SafeLoader)

    def parse(self):
        return self.parsed_yaml_file

