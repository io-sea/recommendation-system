#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Salim Mimouni",
    author_email='salim.mimouni@eviden.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Package that run optimizations for a given workflow and recommend data placements",
    entry_points={
        'console_scripts': [
            'workflow_optimizer=workflow_optimizer.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='workflow_optimizer',
    name='workflow_optimizer',
    packages=find_packages(include=['workflow_optimizer', 'workflow_optimizer.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/audreyr/workflow_optimizer',
    version='0.1.0',
    zip_safe=False,
)
