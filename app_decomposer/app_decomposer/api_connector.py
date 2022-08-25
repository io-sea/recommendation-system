#!/usr/bin/env python
"""
This module proposes classes and functions used to connect to and interact with the Rest API.
"""
from __future__ import division, absolute_import, generators, print_function, unicode_literals, \
    with_statement, nested_scopes  # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2018-2021 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import warnings
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import numpy
from loguru import logger


# Only print out once the InsecureRequestWarning due to the lack of SSL in HTTPS usage.
warnings.filterwarnings("once", category=InsecureRequestWarning)

def upper_char(str_in, list_char=None):
    """
    Transforms in uppercase the characters located at definite index of a string.
    If list_char is None, returns the whole string is set to upper.
    Note that if the index in list_char make no sense (i.e. some values are
    bigger than the length of the string), a warning is issued and the index
    are ignored.

    Args:
        str_in (str): the string to transform.
        list_char (list): the index of the characters that must be transformed,
            given as a list.

    Returns:
        str_out (str): the tranformed string
    """
    # if there is no index (ie list_char is empty):
    if not list_char:
        return str_in.upper()

    # Check that no index in list_char is bigger than the length of the string
    # If it is the case, issue a warning (in future versions, will be done
    # via the logging facility).
    max_index = max(list_char)
    if max_index > len(str_in):
        warnings.warn("Some indexes in list_char are bigger than string length. "
                      "They will be ignored.", UserWarning)
    str_out = str()
    for i, char_i in enumerate(str_in):
        if i in list_char:
            str_out += char_i.upper()
        else:
            str_out += char_i
    return str_out

def get_ok_elements(ref_elem_list, *elem):
    """
    Checks if the set of labels corresponds to the labels available in the reference list
    list_labels.

    Args:
        ref_elem_list (list of strings): the reference list of labels
        *elem (strings): the labels to check in the list.

    Returns:
        (list of strings) the set of incoming labels available in the API. If no labels are
        available return None.
    """
    not_elem = [i for i in elem if i not in ref_elem_list]
    if not_elem:
        warnings.warn("{} not available in the reference list {}".format(not_elem, ref_elem_list),
                      UserWarning)
    ok_elem = set(elem) - set(not_elem)
    return list(ok_elem) if ok_elem else None



class ApiConnect:
    """
    This class proposes an abstract implementation of an API connector.
    """
    def __init__(self, api_uri, api_token, *args, **kwargs):
        """
        Initializes the API connection with the uri and the user token.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
            *args
            **kwargs
        """
        self.uri = api_uri
        self.api_token = api_token
        self.args = args
        self.kwargs = kwargs

    @property
    def api_uri(self):
        """
        Builds the URI to join the API.
        """
        if self.uri.endswith("/"):
            return self.uri + "backend/api/"
        return self.uri + "/backend/api/"

    def build_url(self):
        """
        Generate the url corresponding to all attibutes of the class.
        """
        raise NotImplementedError("The ApiConnect class is abstract and should not be"
                                  " instantiated.")

    def check_connection(self):
        """Check the connection to the server succeeds, by testing it returns the 403 code (bad
        credential) while we perform here a request without token.
        That is the behavior expected for all IOI/IOPA API endpoints, while they are all protected.

        Returns:
            (bool): True if the connection to the server succeeds, False otherwise.
        """
        return check_http_code(self.build_url(), 403)


class TimeSeries(ApiConnect):
    """
    This class allows to retrieve time-series data from the database, connecting to API.
    It uses the method api/details from the API.
    """
    def __init__(self, api_uri, api_token, object_id, type_series,
                 max_point='', export_user='True', start='', end=''):
        """Initializes the TimeSeries class with the api token of the user that access the API, to
        retrieve the time series data of the chosen type and for the chosen job.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
            object_id (string): object id of the considered job.
            type_series (string): the type of time series data to be retrieved (chosen in the
                set of api possibilities).
            max_point (string): number of retrieved data in the time series.
            export_user (string): Raw data or not (default yes).
            start (string): the start time (Unix timestamp).
            end (string): the end time (Unix timestamp).
        """
        super().__init__(api_uri, api_token, object_id, type_series)
        self.object_id = object_id
        self.type_series = type_series
        self.max_point = max_point
        self.export_user = export_user
        self.start = start
        self.end = end
        self.labels = None

    def build_url(self):
        """Generates the url corresponding to all attibutes of the class.

        Returns:
            (string) the api url
        """
        return (self.api_uri
                + 'details?type=' + self.type_series
                + '&objectId=' + self.object_id
                + '&maxPoint=' + self.max_point
                + '&exportUser=' + self.export_user
                + '&start=' + self.start
                + '&end=' + self.end)

    def get_data_by_label(self, *labels):
        """Retrieves, from the database, the data corresponding to the labels.

        Args:
            *labels (strings): the labels of data to be retrieved. Default retrieves all.

        Returns:
            dict: the dictionary of the time-series as numpy arrays of the selected label(s) as
            keys.
        """
        rqst = request_delegator(requests.get, self.build_url(),
                                 headers={'Authorization': self.api_token})
        raw_data = rqst.json()

        array_data = self.numpy_data_format(raw_data)

        dict_data = {}
        for i, lab in enumerate(self.labels):
            dict_data[lab] = array_data[:, i]

        if labels:
            ok_labels = get_ok_elements(self.labels, *labels)
        else:
            ok_labels = self.labels

        try:
            return {i: dict_data[i] for i in ok_labels}
        except TypeError as exc:
            raise ValueError('No labels are available in the reference list. Please verify the'
                             'selected labels availability.') from exc

    def numpy_data_format(self, data):
        """Checks if the format of the raw data, retrieved from the API, is the expected one. If yes
        and labels exist for the current type of time series, returns the data associated.

        Args:
            data (list): list of lists where extract data

        Returns:
            (numpy array) the data of the time-series.
        """
        if not data:
            raise RuntimeError(f'The data is empty: "{data}". Please verify the content of the '
                               f'"{self.object_id}" element.')
        data_array = numpy.asarray(data)  # verify the list is compliant with an array format
        assert len(data_array.shape) == 2, \
            f'Data array "{data_array}" has not uniform dimensions.'
        assert data_array.shape[1] > 0, f'Data array "{data_array}" is empty.'

        self.set_labels(data[0])

        return data_array[1:, :].astype(numpy.int64)

    def set_labels(self, data):
        """Sets the labels of the current type of time series.

        Args:
            data (list): list of lists where the labels should be extracted.
        """
        if not isinstance(data[0], str):
            raise Exception("Warning : No label available for the series "
                            "\"{}\"".format(self.type_series))
        self.labels = data


class MetaData(ApiConnect):
    """
    This class allows to retrieve metadata from the database using the job_id, connecting to API.
    It uses the method api/job/objectid from the API to guarantee the unicity of each job.
    """
    def __init__(self, api_uri, api_token, object_id):
        """Initializes the MetaData class with the api token of the user that access the API, to
        retrieve the metadata for the chosen job.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
            object_id (string): object id of the considered job.
        """
        super().__init__(api_uri, api_token, object_id)
        self.object_id = object_id

    def build_url(self):
        """Generates the url corresponding to all attributes of the class.

        Returns:
            string: the api url
        """
        return (self.api_uri
                + 'job/objectid?'
                + 'objectId=' + self.object_id)

    def get_all_metadata(self):
        """Retrieves, from the database, the metadata corresponding to the job considered.

        Returns:
            dict: the dictionary of the metadata.
        """
        rqst = request_delegator(requests.get, self.build_url(),
                                 headers={'Authorization': self.api_token})
        return rqst.json()

    def get_metadata_by_key(self, *key):
        """Retrieves, from the database, the metadata corresponding to the keys.

        Args:
            *key (strings): the labels of the metadata to be retrieved.

        Returns:
            the sub-dict corresponding to the available key(s) if it exists,
            an empty dict otherwise.
        """
        dict_metadata = self.get_all_metadata()
        all_keys = dict_metadata.keys()
        ok_keys = get_ok_elements(all_keys, *key)
        try:
            return {k: dict_metadata[k] for k in ok_keys}
        except TypeError:
            return dict()


class MinMax(ApiConnect):
    """
    This class allows to retrieve from the database the min and max values for each IOI metric on
    the whole database, connecting to API.
    It uses the method api/details/metrics/minmax from the API.
    """
    def __init__(self, api_uri, api_token):
        """Initializes the MinMax class with the api token of the user that access the API, to
        retrieve the min and max values for the whole set of jobs in the database.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
        """
        super().__init__(api_uri, api_token)

    def build_url(self):
        """Generates the url corresponding to all attributes of the class.

        Returns:
            string: the api url
        """
        return (self.api_uri
                + 'details/metrics/minmax?')

    def get_data(self):
        """Retrieves, from the database, the min and max values of each IOI metric for whole set of
        jobs.

        Returns:
            dict: the dictionary of the min and max values.
        """
        rqst = request_delegator(requests.get, self.build_url(),
                                 headers={'Authorization': self.api_token})
        raw_dict_minmax = rqst.json()

        # Fix the label names that are not yet identical between the api/details (used to retrieve
        # the time-series) method and api/details/metrics/minmax (used here).
        # !!! To be removed when the api method will be updated. !!!
        dict_minmax = dict()
        for label, _ in raw_dict_minmax.items():
            split_lab = label.split('.')
            if 'SbbBbStats' in label and  "sizes" in label:
                for io_range in raw_dict_minmax[label]['max']:
                    new_lab = (split_lab[0].rstrip('sGw')
                               + ''.join([upper_char(sl, [0]) for sl in split_lab[1:]])
                               + upper_char(io_range, [0]))
                    max_range = raw_dict_minmax[label]['max'][io_range]
                    try:
                        min_range = raw_dict_minmax[label]['min'][io_range]
                    except TypeError:
                        min_range = 0
                    dict_minmax[new_lab] = {'max': max_range,
                                            'min': min_range}
            elif "Sizes" in label or "Durations" in label or 'SbbBbStats' in label:
                new_lab = (split_lab[0].rstrip('sGw')
                           + ''.join([upper_char(sl, [0]) for sl in split_lab[1:]]))
                dict_minmax[new_lab] = raw_dict_minmax[label]
            else:
                new_lab = split_lab[1]
                dict_minmax[new_lab] = raw_dict_minmax[label]
        return dict_minmax


class MinMaxDuration(ApiConnect):
    """
    This class allows to retrieve from the database the min and max duration values for all jobs in
    the database. It uses the method api/job/extrem/duration from the API.
    """
    def __init__(self, api_uri, api_token):
        """Initializes the MinMaxDuration class with the api token of the user that access the API,
        to retrieve the min and max duration values for the whole set of jobs in the database.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
        """
        super().__init__(api_uri, api_token)

    def build_url(self):
        """Generates the url corresponding to all attributes of the class.

        Returns:
            string: the api url
        """
        return (self.api_uri
                + 'job/extrem/duration?')

    def get_data(self):
        """Retrieves, from the database, the min and max duration values whole set of jobs.

        Returns:
            dict: the dictionary of the min and max values.
        """
        return request_delegator(requests.get, self.build_url(),
                                 headers={'Authorization': self.api_token}).json()


class LabExpStore(ApiConnect):
    """
    This abstract class allows Optimization/Clustering Lab experiment to send partial or complete
    results to the backend for database storage as a json object.
    """
    def __init__(self, api_uri, api_token, experiment_id):
        """Initializes the LabExpCompleted class with the api token of the user that access
         the API, to post a dictionary.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
            experiment_id (string): the unique experiment id provided by backend at creation.
        """
        super().__init__(api_uri, api_token)
        self.experiment_id = experiment_id if experiment_id is not None else ''

    def build_url(self):
        """Generates the url corresponding to the endpoint.

        Returns:
            None: this method is abstract.
        """
        raise NotImplementedError("The LabExpStore class is abstract and should not be"
                                  " instantiated.")


class LabExpCompleted(LabExpStore):
    """
    This class allows Optimization/Clustering Lab experiment to send complete results to
    the backend for database storage as a json object.
    It uses the endpoint /api/iopa/experiment/completed from the API.
    """
    def __init__(self, api_uri, api_token, experiment_id):
        """Initializes the LabExpCompleted class with the api token of the user that access
         the API, to post a dictionary.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
            experiment_id (string): the unique experiment id provided by backend at creation.
        """
        super().__init__(api_uri, api_token, experiment_id)

    def build_url(self):
        """Generates the url corresponding to the endpoint.

        Returns:
            string: the api url.
        """
        return (self.api_uri
                + 'iopa/'
                + 'experiment/'
                + 'completed'
                + '?objectId=' + self.experiment_id)

    def post_data(self, experiment_data):
        """Builds the request object to be sent to the backend.

        Args:
            experiment_data (dict): experiment data gathered in a dict.
        """
        request_delegator(requests.post, self.build_url(), json=experiment_data,
                          headers={'Authorization': self.api_token})

class JobSearch(ApiConnect):
    """
    This class allows to retrieve from the database the list job metadata according a filter on
    the whole database, connecting to API. If no filter is used, all jobs are retrieved.
    It uses the method api/job/search from the API.
    """
    def __init__(self,
                 api_uri,
                 api_token,
                 job_filter=None,
                 order='',
                 colorder='',
                 limit=0,
                 offset=0):
        """Initializes the JobSearch class with the api token of the user that access the API.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
            job_filter (dict): the filter applied to select the jobs.
            order (string): the type job ordering, ascendant (asc) or descendant (des).
            colorder (string): the column on which apply the ordering.
            limit (int): the limit of selected jobs.
            offset (int): the offset on selected jobs.
        """
        super().__init__(api_uri, api_token)
        self.payload = {
            'filter': job_filter,
            'colorder': colorder,
            'order': order,
            'offset': offset,
            'limit': limit
        }

    def build_url(self):
        """Generates the url corresponding to all attributes of the class.

        Returns:
            string: the api url
        """
        return self.api_uri + 'job/search'

    @property
    def total_jobs(self):
        """
        Returns:
            (int) the total number in the database.
        """
        data = self.get_data()
        return data["totalInDB"]

    @property
    def total_filter(self):
        """
        Returns:
            (int) the total number in the selection.
        """
        data = self.get_data()
        return data["totalFiltered"]

    @property
    def count(self):
        """
        Returns:
            (int) the number of jobs returned.
        """
        data = self.get_data()
        return data["count"]

    @property
    def job_objids_list(self):
        """
        Returns:
            list: the list of the object ids of all jobs in the selection.
        """
        data = self.get_data()
        return [job["id"] for job in data["data"]]

    def get_data(self):
        """Retrieves, from the database, the raw data of the job selection.

        Returns:
            dict: the dictionary of the raw values.
        """
        rqst = request_delegator(requests.post, self.build_url(),
                                 headers={'Authorization': self.api_token},
                                 json=self.payload)

        return rqst.json()


class JobListMultiple(ApiConnect):
    """
    Given a list of jobs object Ids, this class allows to retrieve from the database the metadata
    associated with these jobs, connecting to API. If some object ids do not exist they are
    ignored, such that only existing jobs are returned.
    It uses the method api/job/list/multiple from the API.
    """
    def __init__(self, api_uri, api_token, job_list=''):
        """Initializes the JobListMultiple class with the api token of the user that access the API.

        Args:
            api_uri (string): the uri to join the API.
            api_token (string): the api token of the user account used.
            job_list (list): the list of the jobs object identifiers.
        """
        super().__init__(api_uri, api_token)
        self.raw_job_list = job_list

    def build_url(self):
        """Generates the url corresponding to all attributes of the class.

        Returns:
            string: the api url
        """
        return (self.api_uri
                + 'job/list/multiple?'
                + 'objectIdList=' + ','.join(self.raw_job_list))

    @property
    def count_jobs(self):
        """
        Returns:
            int: the total number of existing jobs.
        """
        data = self.get_data()
        return len(data)

    @property
    def job_list(self):
        """
        Returns:
            list: the list of object ids of existing jobs.
        """
        data = self.get_data()
        return [job['id'] for job in data]

    def get_data(self):
        """Retrieves, from the database, the raw data of all the existing job in the list.

        Returns:
            dict: the dictionary of the raw metadata values.
        """
        rqst = request_delegator(requests.get, self.build_url(),
                                 headers={'Authorization': self.api_token})
        data = rqst.json()
        # If there is no job, the API returns an empty list, not an empty dict, so cast it.
        if not data:
            data = dict()
        return data

def request_delegator(request, *args, verify_ssl=False, **kwargs):
    """Raises exceptions in case of bad given http request.
       Do not verify SSL certificate to allow self-signed certificate.
       In case of HTTPError, warns with response text.

    Args:
        request (requests.request): the request model to be used (e.g. requests.get,
            requests.post, ...).
        verify_ssl (bool): whether the SSL certificates should be verified or not when using HTTPS.
            Default, False.
        *args, **kwargs: the arguments to be passed to the request model.

    Returns:
        request object: the request object if it succeeds, exception otherwise.
    """
    try:
        rqst = request(*args, verify=verify_ssl, **kwargs)
        if rqst.status_code != requests.codes.ok:
            warnings.warn("HTTP request succeeds without standard status code: "
                          "{} with response {}.".format(rqst.status_code, rqst.text),
                          UserWarning)
        rqst.raise_for_status()
        return rqst
    except requests.exceptions.Timeout as ex:
        logger.error("HTTP request timed out: %s", ex)
    except requests.exceptions.ConnectionError as ex:
        logger.error("HTTP request connection error: %s", ex)
    except requests.exceptions.HTTPError as ex:
        logger.error("Bad HTTP request: %s", ex)
    except requests.exceptions.RequestException as ex:
        logger.error("Error: Invalid request: %s", ex)
    except ValueError as ex:
        logger.error("Invalid response content: %s", ex)
    raise RuntimeError("Error: Impossible to connect: {}".format(*args))


def check_http_code(url, code, verify_ssl=False, **kwargs):
    """Check if the server returns the expected http/https code.

    Args:
        url (string): the url to be tested.
        code (int): the expected http/https code returned by the request.
        verify_ssl (bool): whether the SSL certificates should be verified or not when using HTTPS.
            Default, False.
        **kwargs: the arguments to be passed to the request model.

    Returns:
        (bool): True if the expected code is returned by the server, False otherwise.
    """
    try:
        rqst = requests.head(url, verify=verify_ssl, **kwargs)
        return rqst.status_code == code
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return False


