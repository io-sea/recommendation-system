#!/usr/bin/env python
"""
This module define configuration parameters of the iopa-integration test platform.
"""
from __future__ import division, absolute_import, generators, print_function, unicode_literals,\
                       with_statement, nested_scopes # ensure python2.x compatibility

__copyright__ = """
Copyright (C) 2020 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import os
import subprocess
import json
from os.path import join, splitext
from bson.json_util import dumps, loads
from glob import glob
from pymongo import MongoClient

__MONGO_DUMP__ = '/tmp/mongo'


def get_keycloak_token():
    """ Return the authentification keycloak token """

    ioi_backend_url = f"https://{get_keycloak_hostname()}:{get_keycloak_ipv4_port()}" \
                       "/auth/realms/atos-data-management/protocol/openid-connect/token"
    cmd = ['curl', '--silent', '--insecure']
    cmd += ['--header', 'Content-Type: application/x-www-form-urlencoded']
    cmd += ['--request', 'POST', '--data', 'username=ioi-admin', '--data', 'password=password',
            '--data', 'grant_type=password', '--data', 'client_id=io-instrumentation']
    cmd += [ioi_backend_url]

    rc = subprocess.run(cmd, stdout=subprocess.PIPE, shell=False, check=True)
    conf = json.loads(rc.stdout)
    assert('access_token' in conf)

    return conf['access_token']


def get_ioi_container_name():
    """Return the name of the IOI container"""
    return "sdms-node"


def get_backend_hostname():
    """ Return the hostname of the httpd IOI server """

    return "localhost"


def get_backend_ipv4_port():
    """ Return the ipv4 port of the IOI httpd server """

    return 10443


def get_keycloak_hostname():
    """ Return the hostname of the keycloak server """

    return "localhost"


def get_keycloak_ipv4_port():
    """ Return the ipv4 port of the keycloak server """

    return 18443


def get_mongo_hostname():
    """ Return the hostname of the mongo server """

    return "localhost"


def get_mongo_ipv4_port():
    """ Return the ipv4 port of the mongo server """

    return 10017


class MongoDB:
    """This class builds a Mongo connection to the selected database."""
    def __init__(self, host='localhost', port=27017, database='cmdb_database'):
        """Builds a Mongo connection to a database using the hostname and the port of the mongo
        host.
        Several utils are available in the class to dump, restore, drop a database content or a part
        of it.

        Args:
            host (string): the hostname of the mongoDB host.
            port (int): the port of the mongoDB host.
            database (string): the database to connect.
        """
        self.client = MongoClient(host, port)
        self.database = self.client[database]
        self.collections = self.database.list_collection_names()

    def dump(self, dump_dir, collections=None):
        """Dumps, in dump_dir, the content of the collections of the database in a JSON format.
        By default all collections are dumped, but a list of collections can be passed to only dump
        some.

        Args:
            dump_dir (string): the path where the dumped files are stored. Automatically created if
                it does not exist.
            collections (list): the list of collection to be dumped. Defaults, all collections of
                the database are dumped.
        """
        # Create the dump directory if it does not exist
        os.makedirs(dump_dir, exist_ok=True)

        # Set the collections to be dumped
        dump_collections = self.collections
        if collections:
            dump_collections = collections

        # Dump the collections in JSON files
        for collection in dump_collections:
            jsonpath = join(dump_dir, collection + ".json")
            with open(jsonpath, 'w') as jsonfile:
                jsonfile.write(dumps(self.database[collection].find()))

    def restore(self, dump_dir, drop=False):
        """Restores, from dump_dir, the content of some collections (previously stored in JSON
        format) in the database.
        All collections in dump_dir are restored.

        Args:
            dump_dir (string): the path where the dumped files are stored.
            drop (bool): if the mongoDB collections are erased or not, before restoring its.
                Defaults, the collections are not erased.
        """
        # Restore all the saved collection content
        files = glob(join(dump_dir, "*.json"))
        for file in files:
            collection = splitext(file)[0].split('/')[-1]
            # Erase the collections content if the drop option is activated
            if drop:
                self.database.drop_collection(collection)
            with open(file, 'r') as jsonfile:
                file_data = loads(jsonfile.read())
            if file_data:
                self.database[collection].insert_many(file_data)

    def drop_db(self):
        """Erases all the content of the database (all collections)."""
        self.client.drop_database(self.database.name)
