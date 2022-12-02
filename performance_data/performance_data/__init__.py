"""Top-level package for performance_data."""
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
__author__ = """Salim Mimouni"""
__email__ = 'salim.mimouni@atos.net'
__version__ = '0.1.0'

# dataset file name for performance model
DATASET_FILE = os.path.join(CURRENT_DIR, "dataset",
                            "performance_model_dataset_complete.csv")
