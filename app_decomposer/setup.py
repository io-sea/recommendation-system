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
    author_email='salim.mimouni@atos.net',
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
    description="Transform IOI timeseries traces into a sequential representation of an HPC application",
    entry_points={
        'console_scripts': [
            'app_decomposer=app_decomposer.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='app_decomposer',
    name='app_decomposer',
    packages=find_packages(include=['app_decomposer', 'app_decomposer.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/physics-programmer/app_decomposer',
    version='0.1.0',
    zip_safe=False,
)
