"""Top-level package for app_decomposer."""
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEFAULT_CONFIGURATION = os.path.join(CURRENT_DIR, "defaults", "config.yaml")
KIWI_CONFIG = os.path.join(os.path.dirname(CURRENT_DIR), "tests_integration", "test_data", "test_kiwi_config.yaml")

__author__ = """Salim Mimouni"""
__email__ = 'salim.mimouni@atos.net'
__version__ = '0.1.0'


"""Init for datafetchers.
Sets the default configuration as global variables.
"""

# The API_DICT_TS structure represents how the time-series are reachable through the API. Each
# time-series is accessed by its group (keys in the dictionary API_DICT_TS).
# This is the exhaustive reference list of available time-series in the database.
# Note, in the future, this hardcoded structure should be dynamically retrieved from the API using
# a dedicated method (to be designed).
API_DICT_TS = {
    'volume': ['bytesRead', 'bytesWritten'],
    'processCount': ['ioActiveProcessCount', 'processCount'],
    'ioEfficiencyDistribution': ['accelerated', 'accelPossible', 'accelImpossible'],
    'accessPattern': ['accessRandRead', 'accessSeqRead', 'accessStrRead', 'accessUnclRead',
                      'accessRandWrite', 'accessSeqWrite', 'accessStrWrite', 'accessUnclWrite'],
    'files': ['filesCreated', 'filesDeleted', 'filesRO', 'filesWO', 'filesRW'],
    'ioSizes': ['IOSizeReadRange0', 'IOSizeReadRange1', 'IOSizeReadRange2', 'IOSizeReadRange3',
                'IOSizeReadRange4', 'IOSizeReadRange5', 'IOSizeReadRange6', 'IOSizeWriteRange0',
                'IOSizeWriteRange1', 'IOSizeWriteRange2', 'IOSizeWriteRange3', 'IOSizeWriteRange4',
                'IOSizeWriteRange5', 'IOSizeWriteRange6'],
    'ioDurations': ['IODurationReadRange0', 'IODurationReadRange1', 'IODurationReadRange2',
                    'IODurationReadRange3', 'IODurationReadRange4', 'IODurationReadRange5',
                    'IODurationReadRange6', 'IODurationReadRange7', 'IODurationReadRange8',
                    'IODurationReadRange9', 'IODurationReadRange10', 'IODurationReadRange11',
                    'IODurationReadRange12', 'IODurationReadRange13', 'IODurationReadRange14',
                    'IODurationReadRange15', 'IODurationWriteRange0', 'IODurationWriteRange1',
                    'IODurationWriteRange2', 'IODurationWriteRange3', 'IODurationWriteRange4',
                    'IODurationWriteRange5', 'IODurationWriteRange6', 'IODurationWriteRange7',
                    'IODurationWriteRange8', 'IODurationWriteRange9', 'IODurationWriteRange10',
                    'IODurationWriteRange11', 'IODurationWriteRange12', 'IODurationWriteRange13',
                    'IODurationWriteRange14', 'IODurationWriteRange15'],
    'operationsCount': ['operationRead', 'operationWrite']
    }
# sampling period of IO Instrumentation
IOI_SAMPLING_PERIOD = 5 # in seconds

# dataset file name for performance model
DATASET_SOURCE = os.path.join(REPO_DIR, "performance_data",
                              "performance_data", "dataset",
                              "performance_model_dataset.csv")

