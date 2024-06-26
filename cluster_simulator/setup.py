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
    author="Data Management",
    author_email='bds-datamanagement@atos.net',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A package that simulates  I/O application behavior in multi-tiered HPC cluster system",
    entry_points={
        'console_scripts': [
            'cluster_simulator=cluster_simulator.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cluster_simulator',
    name='cluster_simulator',
    packages=find_packages(include=['cluster_simulator', 'cluster_simulator.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.jsc.fz-juelich.de/io-sea/io-sea-3.4-analytics/cluster_simulator',
    version='0.1.0',
    zip_safe=False,
)
