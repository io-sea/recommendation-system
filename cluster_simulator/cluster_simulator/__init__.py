"""Top-level package for Cluster Simulator."""
import os
__email__ = 'bds-datamanagement@atos.net'
__version__ = '0.1.0'

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(CURRENT_DIR, "defaults", "cluster_config.yaml")
