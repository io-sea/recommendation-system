#!/usr/bin/env python
""" This module allows the connection to a mongo database to retrieve data metrics of jobs"""

from __future__ import division, absolute_import, generators, print_function, unicode_literals,\
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2017 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import itertools
import numpy as np
import pandas as pd
from ioanalyticstools import string_manipulation, normalizers

IO_DURATIONS_HISTO_BIN = 16
IO_SIZES_HISTO_BIN = 7


class MetricsDataBuilder:
    """ Build a matrix of a selection of feature, in a normalized version, for a specific jobid """
    def __init__(self, dict_db, jobid):
        if dict_db['JobItem'].loc[jobid].empty:
            raise Exception('There is not such jobid in this database:', jobid)
        self.dict_db = dict_db
        self.jobid = jobid

    def select_data_from_db(self, selected_metrics):
        """Collect a set of selected metrics from the database dictionnary, and store them in a
        pandas dataframe

        Args:
            selected_metrics : a dictionary containing the collection names as keys and a list of
            attributes to extract as values

        Returns:
            a pandas dataframe
        """
        data = pd.DataFrame()
        for collection, attributes in selected_metrics.items():
            for attr in attributes:
                data[attr] = self.dict_db[collection].loc[self.jobid][attr]
        return data

    def select_metadata_from_db(self, selected_metadata):
        """Collect a set of selected metadatas from the database dictionary, and store them in a
        dict

        Args:
            selected_metadata : a dictionary containing the collection names as keys and a list of
            attributes to extract as values

        Returns:
            a dict
        """
        data = dict()
        for collection, attributes in selected_metadata.items():
            for attr in attributes:
                data[attr] = self.dict_db[collection].loc[self.jobid][attr]
        return data

    def normalize_access_pattern(self, data):
        """ Transform a data frame with raw values in a normalized data frame

        Args:
            raw data frame

        Returns:
            the scaled data frame
        """
        data = pd.DataFrame()

        # FileIOSummaryGw
        fiogw = self.dict_db['FileIOSummaryGw']
        label_not_norm = ["accessRandRead",
                          "accessSeqRead",
                          "accessStrRead",
                          "accessUnclRead",
                          "accessRandWrite",
                          "accessSeqWrite",
                          "accessStrWrite",
                          "accessUnclWrite"]

        ratio_pattern_r = pd.DataFrame(fiogw.loc[self.jobid][["accessRandRead",
                                                              "accessSeqRead",
                                                              "accessStrRead",
                                                              "accessUnclRead"]])

        ratio_pattern_w = pd.DataFrame(fiogw.loc[self.jobid][["accessRandWrite",
                                                              "accessSeqWrite",
                                                              "accessStrWrite",
                                                              "accessUnclWrite"]])

        for label in list(label_not_norm):
            if "Read" in label:
                data[label] = (ratio_pattern_r[label]/(ratio_pattern_r.sum(axis=1))).fillna(0)
            else:
                data[label] = (ratio_pattern_w[label]/(ratio_pattern_w.sum(axis=1))).fillna(0)

        return data

    def get_full_histogram(self, collection, mode, norm=""):
        """Extract data values of an histogram in the database

        Args:
            the name of the collection in the db
            the version of the histogram (read or write)
            normalization method (meanmax, zscore, or no by default)

        Returns:
            a panda dataframe of bins the histogram values (included zero values) for each timeframe
            a panda series of the histogram values for the whole job (sum along all the timeframe)
        """
        if collection == 'IODurationsGw':
            n_bin_histo = IO_DURATIONS_HISTO_BIN
        elif collection == 'IOSizesGw':
            n_bin_histo = IO_SIZES_HISTO_BIN
        else:
            raise NameError('Unknown collection name')

        # Build the histogram for all timeframes
        dict_histo = self.dict_db[collection].loc[self.jobid][mode]
        label_ranges = ['range'+str(i) for i in range(n_bin_histo)]
        number_timeframe = len(dict_histo.index)
        list_histo = np.empty(shape=(0, n_bin_histo))
        idx_jid = [self.jobid]*number_timeframe
        idx_tf = dict_histo.keys()
        for _, tframe in dict_histo.iteritems():
            histogram = [tframe[label_ranges[i]] if label_ranges[i] in tframe.keys() else 0\
                         for i in range(n_bin_histo)]
            list_histo = np.append(list_histo, np.array(histogram).reshape(1, -1), axis=0)
        names = self.dict_db[collection].index.names
        index = pd.MultiIndex.from_tuples(list(zip(*[idx_jid, idx_tf])), names=names)

        # Build the histogram of the job (sum along all timeframe)
        histo_job_name = collection+string_manipulation.upper_char(mode, [0])
        list_histo_job = list_histo.sum(0)

        # Normalize histogram
        if norm == "meanmax":
            list_histo_job = normalizers.meanmax_normalization(list_histo_job)
        elif norm == "zscore":
            list_histo_job = normalizers.standard_normalization(list_histo_job)

        return pd.DataFrame(list_histo, index=index), pd.Series([list_histo_job],
                                                                index=[self.jobid],
                                                                name=histo_job_name)

    def collect_histograms_from_db(self, collections, modes, norm=""):
        """Collect all the combination of collections/modes histograms from the database

        Args:
            the name of the collections in the db
            the versions of the histogram (read and/or write)
            normalization method

        Returns:
            a dataframe of the 4 histograms in the order (IODurationsRead/IODurationsWrite
            /IOSizesRead/IOSizesWrite)
        """
        dict_series = {}
        for col, mode in list(itertools.product(collections, modes)):
            _, histo_job = self.get_full_histogram(col, mode, norm)
            dict_series[histo_job.name] = histo_job
        return pd.DataFrame(dict_series)

    def get_job_histograms(self, norm=""):
        """Collect the 4 histograms of a job from the database

        Args:
            normalization method

        Returns:
            a dataframe of the 4 histograms in the order (IODurationsRead/IODurationsWrite/
            IOSizesRead/IOSizesWrite)
        """
        collections = ["IODurationsGw", "IOSizesGw"]
        modes = ["read", "write"]
        return self.collect_histograms_from_db(collections, modes, norm)

    def get_version(self):
        """Get the version of IOI corresponding to the data in base for the current jobid

        Returns:
            a 3 digit number (major/minor/release)
        """
        str_version = self.dict_db['JobItem'].loc[self.jobid].version
        return int(''.join([i for i in str_version if i.isdigit()]))
