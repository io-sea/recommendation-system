"""Top-level package for performance_data."""
import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
__email__ = 'bds-datamanagement@atos.net'
__version__ = '0.1.0'

# dataset file name for performance model
DATASET_FILE = os.path.join(CURRENT_DIR, "dataset",
                            "performance_model_dataset_complete.csv")

GENERATED_DATASET_FILE = os.path.join(os.path.dirname(CURRENT_DIR),
                                      "tests",  "test_data",
                                      "complete_test_generated_dataset.csv")
# directory where the models are saved
MODELS_DIRECTORY = os.path.join(os.path.dirname(CURRENT_DIR),
                                "defaults", "models")
