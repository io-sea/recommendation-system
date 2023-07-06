"""This module provides unit testing for the APIConnector.
"""
import unittest
import requests
import requests_mock
from pydantic.error_wrappers import ValidationError
from app_decomposer.connector import APIConnector
from app_decomposer.connector import ApiModel


class TestInputModel(ApiModel):
    value_1: str
    value_2: int


class TestOutputModel(ApiModel):
    value_1: str
    value_2: int


class TestConnector(unittest.TestCase):

    def setUp(self) -> None:
        """Set up the unit tests by creating an object of class api_connector without mock enbled
        and an object of class api_connector with mock enabled.
        """
        self.api_connector = APIConnector(api_uri="http://test_uri/",
                                          api_token="test_token")
        self.api_connector_mocked = APIConnector(api_uri="http://test_uri",
                                                 api_token="test_token",
                                                 mock=True)

    def test_prepare_request(self):
        """Test that the request path is correct, when giving a path and params.
        """
        test_request = self.api_connector._prepare_request(method="GET",
                                                           endpoint="/test",
                                                           params={"value_1": "1", "value_2": "2"})
        self.assertEqual(test_request.url,
                         "http://test_uri/test?value_1=1&value_2=2")

    def test_prepare_request_input_model(self):
        """Test that the request delegator works as expected when there is
        an output model.
        """
        self.api_connector._prepare_request(method="GET",
                                            endpoint="/test",
                                            input_json={
                                                "value_1": "test", "value_2": 3},
                                            input_model=TestInputModel)

    def test_prepare_request_input_model_fail(self):
        """Tests that the request delegator behaves as expected when the input model does not fit
        the expected input model by adding an extra field.
        """
        with self.assertRaises(ValidationError) as ex:
            self.api_connector._prepare_request(method="GET",
                                                endpoint="/test",
                                                input_json={"value_1": "test",
                                                            "value_2": 3, "value_3": "unknown"},
                                                input_model=TestInputModel)
            self.assertIn(ex.exception, "extra fields not permitted")

    def test_prepare_request_input_model_type_fail(self):
        """Tests that the request delegator behaves as expected when the input model does not fit
        the expected input model by giving the wrong type.
        """
        with self.assertRaises(ValidationError) as ex:
            self.api_connector._prepare_request(method="GET",
                                                endpoint="/test",
                                                input_json={"value_1": "test",
                                                            "value_2": "test"},
                                                input_model=TestInputModel)
            self.assertIn(ex.exception, "value is not a valid integer")

    @requests_mock.Mocker()
    def test_request_delegator_output_model(self, mocker):
        """Tests the request delegator when there is an output model to deserialize the model.
        """
        response_json = {
            "value_1": "test",
            "value_2": 2
        }
        mocker.get("http://test_uri/test", json=response_json)
        response = self.api_connector.request_delegator(method="GET",
                                                        endpoint="/test",
                                                        output_model=TestOutputModel)
        self.assertEqual(response.parsed_json.value_1, "test")
        self.assertEqual(response.parsed_json.value_2, 2)

    @requests_mock.Mocker()
    def test_request_delegator_output_model_type_fail(self, mocker):
        """Tests the request delegator when there is an output model to deserialize the model and
        the type of the answer does not match the expected output model.
        """
        response_json = {
            "value_1": "test",
            "value_2": "test"
        }
        mocker.get("http://test_uri/test", json=response_json)
        with self.assertRaises(ValueError):
            self.api_connector.request_delegator(method="GET",
                                                 endpoint="/test",
                                                 output_model=TestOutputModel)

    @requests_mock.Mocker()
    def test_request_delegator_output_model_extra_attribute(self, mocker):
        """Tests the request delegator when there is an output model to deserialize the model and
        there is an extra value that does not match the output model.
        """
        response_json = {
            "value_1": "test",
            "value_2": 3,
            "value_3": "test"
        }
        mocker.get("http://test_uri/test", json=response_json)
        with self.assertRaises(ValueError):
            self.api_connector.request_delegator(method="GET",
                                                 endpoint="/test",
                                                 output_model=TestOutputModel)

    @requests_mock.Mocker()
    def test_check_connection(self, mocker):
        """Tests that checking the connection works as expected:
        - when there is a 403 error code it returns true
        - when there is 404 error code it returns false
        - when there is a timeout or a connection error, it returns false.
        a 404."""
        mocker.get("http://test_uri/tools/version", status_code=200)
        self.assertTrue(self.api_connector.check_connection())
        mocker.get("http://test_uri/tools/version", status_code=404)
        self.assertFalse(self.api_connector.check_connection())
        mocker.get("http://test_uri/tools/version",
                   exc=requests.exceptions.ConnectTimeout)
        self.assertFalse(self.api_connector.check_connection())
        mocker.get("http://test_uri/tools/version",
                   exc=requests.exceptions.ConnectionError)
        self.assertFalse(self.api_connector.check_connection())

    @requests_mock.Mocker()
    def test_request_delegator_error_codes(self, mocker):
        """Tests that the request delegator works as expected whenever getting different exceptions:
        - Timeouts (tested through mocking)
        - Connection error (tested through mocking)
        - HTTP error (tested through mocking)
        - Request Exception (tested through mocking)
        - Value error (tested through mocking)
        """
        mocker.get("http://test_uri/",
                   exc=requests.exceptions.ConnectTimeout)
        with self.assertRaises(requests.exceptions.ConnectTimeout):
            self.api_connector.request_delegator(method="GET", endpoint="")

        mocker.get("http://test_uri/",
                   exc=requests.exceptions.ConnectionError)
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.api_connector.request_delegator(method="GET", endpoint="")

        mocker.get("http://test_uri/",
                   exc=requests.exceptions.HTTPError)
        with self.assertRaises(requests.exceptions.HTTPError):
            self.api_connector.request_delegator(method="GET", endpoint="")

        mocker.get("http://test_uri/",
                   exc=requests.exceptions.RequestException)
        with self.assertRaises(requests.exceptions.RequestException):
            self.api_connector.request_delegator(method="GET", endpoint="")

        mocker.get("http://test_uri/",
                   exc=ValueError)
        with self.assertRaises(ValueError):
            self.api_connector.request_delegator(method="GET", endpoint="")


if __name__ == "__main__":
    unittest.main()
