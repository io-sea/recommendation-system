"""This module describes the APIConnector, a general module to connect to an API, in order
to perform data exchange through HTTP requests.
"""

__copyright__ = """
Copyright (C) Bull S. A. S. - All rights reserved
"""
from typing import Any, Dict, Type
from pydantic.error_wrappers import ValidationError
from pydantic import BaseModel, Extra

import requests
import warnings
from loguru import logger
import requests
from tests_integration.iopa_integration_config import get_keycloak_token


class Connector:
    def __init__(self, base_url):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {get_keycloak_token()}",
            "Content-Type": "application/json",
        }

    def get(self, endpoint):
        response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def post(self, endpoint, data):
        response = requests.post(f"{self.base_url}/{endpoint}", headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()


def get_keycloak_token_from_config_file(get_keycloak_hostname, get_keycloak_ipv4_port, client_id, username='smchpcadmin', password='smchpcadmin'):
    """ Return the authentication keycloak token """

    ioi_backend_url = f"https://{get_keycloak_hostname}:{get_keycloak_ipv4_port}" \
        "/auth/realms/atos-data-management/protocol/openid-connect/token"
    cmd = ['curl', '--silent', '--insecure']
    cmd += ['--header', 'Content-Type: application/x-www-form-urlencoded']
    cmd += ['--request', 'POST', '--data', f'username={username}', '--data', f'password={password}',
            '--data', 'grant_type=password', '--data', f'client_id={client_id}']
    cmd += [ioi_backend_url]
    rc = subprocess.run(cmd, stdout=subprocess.PIPE, shell=False, check=True)
    print(f"Output: {rc.stdout}") 
    conf = json.loads(rc.stdout)
    assert ('access_token' in conf)

    return conf['access_token']


class ApiModel(BaseModel):
    """Base class for all schema describing the data exchanged across all API calls, in order to
    have a common configuration across all different models.
    """
    class Config:
        """Define the configuration for the pydantic models.

        For now:
            - Raise a validation error when an extra field is supplied
        """
        extra = Extra.forbid


class APIConnector:
    """Generic class for connection to the API.
    """

    def __init__(self,
                 api_uri: str,
                 api_token: str,
                 verify_ssl: bool = False,
                 mock: bool = False,
                 suffix: str = "") -> None:
        """Initialization of a generic class to connect to the API, by specifying the different
        information for reaching the API.

        Args:
            api_uri (str): The URI to the API.
            api_token (str): The identification token to log into the API.
            verify_ssl (bool, optional): Whether or not to check for SSL certificates.
                Defaults to False.
            mock (bool, optional): Whether or not to use a mock API which mocks HTTP calls.
                Defaults to False.
            suffix (str, optional): The suffix appended to the API uri. Defaults to an empty
                string.
        """
        self.api_uri = (api_uri if not api_uri.endswith(
            "/") else api_uri[:-1]) + suffix
        self.api_token = api_token
        self.verify_ssl = verify_ssl
        # if mock:
        #     # If mock is enabled, use FastAPI (starlette) TestClient in order to mock HTTP calls
        #     self.session = TestClient(app)
        # else:
        #     self.session = requests.Session()
        self.session = requests.Session()

    def _prepare_request(self,
                         method: str,
                         endpoint: str,
                         params: Dict[str, Any] = None,
                         input_json: Dict[str, Any] = None,
                         input_model: Type[ApiModel] = None,
                         ) -> requests.PreparedRequest:
        """Prepare an object of type requests.Request, which will then be used for sending the
        requests to the API.

        Args:
            method (str): the method used on the endpoint.
                Can either be: post, put, delete, patch
            endpoint (str): The path to the endpoint.
            params (Dict[str, Any], optional): The parameters to give to the request.
                Defaults to None (no parameters sent with the request).
            input_json (Dict[str, Any], optional): The dictionary of the json sent with the request.
                Defaults to None (no json sent with the request).
            input_model (BaseModel, optional): A Pydantic model to validate the input json with.
                Defaults to None (no validation performed on the input data).

        Returns:
            requests.PreparedRequest: An object of type PreparedRequest.
        """
        if input_model and input_json is not None:
            input_json = input_model(**input_json).dict(exclude_none=True)
        return requests.Request(method=method,
                                url=f"{self.api_uri}{endpoint}",
                                params=params,
                                json=input_json,
                                headers={'Authorization': self.api_token}).prepare()

    def request_delegator(self,
                          method: str,
                          endpoint: str,
                          params: Dict[str, Any] = None,
                          input_json: Dict[str, Any] = None,
                          input_model: Type[ApiModel] = None,
                          output_model: Type[ApiModel] = None,
                          ) -> requests.Response:
        """Send a request to the desired endpoint, with the wanted params and input_json,
        with the possibility to validate the input data using a Pydantic model, as well as
        the output data, by adding an attribute parsed_json as a Pydantic model to the result of the
        request.

        Args:
            method (str): the method used on the endpoint.
                Can either be: post, put, delete, patch
            endpoint (str): The path to the endpoint.
            params (Dict[str, Any], optional): The parameters to give to the request.
                Defaults to None (no parameters sent with the request).
            input_json (Dict[str, Any], optional): The dictionary of the json sent with the request.
                Defaults to None (no json sent with the request).
            input_model (BaseModel, optional): A Pydantic model to validate the input json with.
                Defaults to None (no validation performed on the input data).
            output_model (BaseModel, optional): A Pydantic model describing the expected output
                data. If set to a value, there will be an additional attribute parsed_json to the
                output reponse, corresponding to the pydantic model.
                Defaults to None (no validation performed on the output data)

        Returns:
            Response: the response of the sent request
        """
        try:
            logger.debug("Sending %s request to endpoint %s", method, endpoint)
            prepared_request = self._prepare_request(method,
                                                     endpoint,
                                                     params,
                                                     input_json,
                                                     input_model)
            with self.session as session:
                response = session.send(
                    prepared_request, verify=self.verify_ssl)
                if response.status_code != requests.codes.ok:
                    warnings.warn("HTTP request succeeds without standard status code: "
                                  "{} with response {}.".format(
                                      response.status_code, response.text),
                                  UserWarning)
                response.raise_for_status()
                if output_model:
                    try:
                        response.parsed_json = output_model(**response.json())
                    except ValidationError as err:
                        raise ValueError("Invalid response data format") from err
                return response
        except requests.exceptions.Timeout as ex:
            logger.error("HTTP request timed out: %s", ex)
            raise ex
        except requests.exceptions.ConnectionError as ex:
            logger.error("HTTP request connection error: %s", ex)
            raise ex
        except requests.exceptions.HTTPError as ex:
            logger.error("Bad HTTP request: %s", ex)
            raise ex
        except requests.exceptions.RequestException as ex:
            logger.error("Error: Invalid request: %s", ex)
            raise ex
        except ValueError as ex:
            logger.error("Invalid response content: %s", ex)
            raise ex

    def check_connection(self) -> bool:
        """Check the connection to the server succeeds, by testing it returns the 200 code
        when accessing the /version endpoint.

        Returns:
            bool: whether or not the /version endpoint returns a 200 success code.
        """
        return self.check_http_code(f"{self.api_uri}/tools/version", 200)

    def check_http_code(self,
                        url: str,
                        code: int,
                        ** kwargs) -> bool:
        """Check if the server returns the expected http/https code.

          Args:
              url (string): the url to be tested.
              code (int): the expected http/https code returned by the request.
              verify_ssl (bool): whether the SSL certificates should be verified or not when using HTTPS.
                  Default, False.
              **kwargs: the arguments to be passed to the request model.

          Returns:
              bool: True if the expected code is returned by the server, False otherwise.
          """
        try:
            with self.session as session:
                rqst = session.get(url, verify=self.verify_ssl)
            return rqst.status_code == code
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            return False
