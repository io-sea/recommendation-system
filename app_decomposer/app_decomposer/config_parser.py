"""
This module parses a YAML configuration file.
"""

__copyright__ = """
Copyright(C) 2022 Bull S. A. S. - All rights reserved
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
        self.parsed_yaml_file = yaml.load(Path(path).read_text(), 
                                          Loader=yaml.SafeLoader)

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
    
    def get_proxy(self):
        """Parse the YAML configuration to get proxy settings."""
        proxy_config = self.parsed_yaml_file.get("proxy", {})
        http_proxy = proxy_config.get("http", None)
        https_proxy = proxy_config.get("https", None)

        if http_proxy or https_proxy:
            return {
                "http": http_proxy,
                "https": https_proxy
            }
        return {}

    def get_kc_token(self):
        """Get a keycloak valid token."""
        ioi_backend_url = f"{self.get_kc_hostname()}:{self.get_kc_port()}"\
            f"/auth/realms/{self.get_kc_realm()}/protocol/openid-connect/token"
        cmd = ['curl', '-v', '--silent', '--insecure']

        proxy = self.get_proxy().get('http')  # Just get the HTTP proxy
        if proxy:
            cmd += ['--proxy', proxy]

        cmd += ['--header', 'Content-Type: application/x-www-form-urlencoded']
        cmd += ['--request', 'POST', '--data', 'username=ioi-admin', '--data', 'password=password',
                '--data', 'grant_type=password', '--data', f'client_id={self.parse()["keycloak"]["client"]}']

        cmd += [ioi_backend_url]
        print(f"CMD = {cmd}")
        #rc = subprocess.run(cmd, stdout=subprocess.PIPE, shell=False, check=True)
        rc = subprocess.run(cmd, shell=False, check=True)
        conf = json.loads(rc.stdout)
        assert('access_token' in conf)
        return conf["access_token"]


    
    def get_kc_token_docker(self):
        """Get a keycloak valid token."""
        ioi_backend_url = f"{self.get_kc_hostname()}:{self.get_kc_port()}"\
            f"/auth/realms/{self.get_kc_realm()}/protocol/openid-connect/token"            
        cmd = ['curl', '--silent', '--insecure']
        cmd += ['--header', 'Content-Type: application/x-www-form-urlencoded']
        cmd += ['--request', 'POST', '--data', 'username=bird_admin', '--data', 'password=bird_admin',
                '--data', 'grant_type=password', '--data', f'client_id={self.parse()["keycloak"]["client"]}']

        cmd += [ioi_backend_url]
        rc = subprocess.run(cmd, stdout=subprocess.PIPE, shell=False, check=True)
        conf = json.loads(rc.stdout)
        assert('access_token' in conf)

        # if "Bearer " not in conf['access_token']:
        #     token = "Bearer " + conf["access_token"]
        return conf["access_token"]
    
    def get_kc_token_v5(self, username='smchpcadmin', 
                        password='smchpcadmin'):
        """Get a keycloak valid token."""
        ioi_backend_url = f"{self.get_kc_hostname()}:{self.get_kc_port()}"\
            f"/auth/realms/{self.get_kc_realm()}/protocol/openid-connect/token"
        #cmd = ['curl', '--silent', '--insecure']
        cmd = ['curl', '--location', '--request']
        cmd += ['--header', 'Content-Type: application/x-www-form-urlencoded']
        cmd += ['--request', 'POST', '--data', f"username={username}", '--data',
                f"password={password}",
                '--data', 'grant_type=password', '--data', f'client_id={self.parse()["keycloak"]["client"]}']

        cmd += [ioi_backend_url]
        print(f"ioi_backend_url: {ioi_backend_url}")
        rc = subprocess.run(cmd, stdout=subprocess.PIPE, shell=False, check=True)
        # print(f"rc.stdout: {rc.stdout}")
        conf = json.loads(rc.stdout)
        assert('access_token' in conf)

        # if "Bearer " not in conf['access_token']:
        #     token = "Bearer " + conf["access_token"]
        return conf["access_token"]


