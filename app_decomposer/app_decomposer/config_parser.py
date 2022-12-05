"""
This module parses a YAML configuration file.
"""

__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
""" = """
Copyright (C) 2021 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

from pathlib import Path
import yaml
import subprocess
import json
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
        """Returns all yaml content as a dict."""
        return self.parsed_yaml_file

    def get_kc_hostname(self):
        """Get the keycloak hostname."""
        return self.parse()["keycloak"]["uri"]

    def get_kc_port(self):
        """Get the keycloak port."""
        return self.parse()["keycloak"]["port"]

    def get_kc_realm(self):
        """Get the keycloak realm."""
        return self.parse()["keycloak"]["realm"]

    def get_api_uri(self):
        """Get the api uri."""
        return self.parse()["api"]["uri"]

    def get_api_port(self):
        """Get api port."""
        return 10443 if "port" not in self.parse()["api"] else self.parse()["api"]["port"]

    def get_kc_token(self):
        """Get a keycloak valid token."""
        ioi_backend_url = f"{self.get_kc_hostname()}:{self.get_kc_port()}"\
            f"/auth/realms/{self.get_kc_realm()}/protocol/openid-connect/token"
            #f"/auth/realms/{self.get_kc_realm()}/protocol/openid-connect/token"
        cmd = ['curl', '--silent', '--insecure']
        cmd += ['--header', 'Content-Type: application/x-www-form-urlencoded']
        cmd += ['--request', 'POST', '--data', 'username=ioi-admin', '--data', 'password=password',
                '--data', 'grant_type=password', '--data', f'client_id={self.parse()["keycloak"]["client"]}']

        cmd += [ioi_backend_url]
        print(cmd)
        rc = subprocess.run(cmd, stdout=subprocess.PIPE, shell=False, check=True)
        conf = json.loads(rc.stdout)
        assert('access_token' in conf)

        # if "Bearer " not in conf['access_token']:
        #     token = "Bearer " + conf["access_token"]
        return conf["access_token"]


