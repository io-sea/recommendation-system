"""
Setup file for the jobmodel module.
"""

#!/usr/bin/env python

__copyright__ = """
Copyright (C) 2018 Bull S. A. S. - All rights reserved
Bull, Rue Jean Jaures, B.P.68, 78340, Les Clayes-sous-Bois, France
This is not Free or Open Source software.
Please contact Bull S. A. S. for details about its license.
"""

import os
import glob
from setuptools import setup, find_packages

WHEEL_VERSION = '2.0'
VERSION = os.environ['RPM_VERSION'] if 'RPM_VERSION' in os.environ else WHEEL_VERSION


def readme():
    """
    Return content of README file as string
    """
    if glob.glob("README.md"):
        with open('README.md', encoding="utf-8") as readmefile:
            return readmefile.read()
    else:
        return ""


def requirements(req_path):
    """
    List requirements from requirements.txt file
    """
    if glob.glob(req_path):
        with open(req_path) as requirements_file:
            return [req.strip() for req in requirements_file.readlines()]
    else:
        return []


setup(
    name='jobmodel',
    version=VERSION,
    url='',
    license='',
    description='Python package to build model of jobs',
    long_description=readme(),
    author='BDS Data Management',
    author_email='bds-datamanagement@atos.net',
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements("requirements.txt"),
    extras_require={
        'test': requirements(os.path.join("tests", "test_requirements.txt"))
    },
    package_data={"": ["*.pyc"]}
    )
