import os
import json
import pandas as pd
import re
from loguru import logger


def camel_case_to_snake_case(name):
    """
    Convert a string from CamelCase to snake_case.
    """
    logger.debug("Converting CamelCase to snake_case")
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)).lower()

def get_column_names(json_file_name):
    """
    Given a JSON file name, this function returns a list of relevant column names.
    """
    logger.debug(f"Getting column names for {json_file_name}")
    prefix = camel_case_to_snake_case(json_file_name.split('.')[0])
    column_name_mapping = {
        "accessPatternRead.json": ["timestamp", f"{prefix}_random", f"{prefix}_sequential", f"{prefix}_stride", f"{prefix}_unclassified"],
        "accessPatternWrite.json": ["timestamp", f"{prefix}_random", f"{prefix}_sequential", f"{prefix}_stride", f"{prefix}_unclassified"],
        "ioSizesRead.json": ["timestamp", f"{prefix}_0B_16B", f"{prefix}_16B_4KB", f"{prefix}_4KB_128KB", f"{prefix}_128KB_1MB", f"{prefix}_1MB_16MB", f"{prefix}_16MB_128MB", f"{prefix}_128MB_+"],
        "ioSizesWrite.json": ["timestamp", f"{prefix}_0B_16B", f"{prefix}_16B_4KB", f"{prefix}_4KB_128KB", f"{prefix}_128KB_1MB", f"{prefix}_1MB_16MB", f"{prefix}_16MB_128MB", f"{prefix}_128MB_+"],
        "operationsCount.json": ["timestamp", f"{prefix}_read", f"{prefix}_write"],
        "volume.json": ["timestamp", "bytesRead", "bytesWritten"]
    }
    return column_name_mapping.get(json_file_name, [])

def is_file_extension(filename, expected_extension):
    """
    Check if the file has the expected extension.
    """
    logger.debug(f"Checking file extension for {filename}")
    _, file_extension = os.path.splitext(filename)
    return file_extension == f".{expected_extension}"

def list_and_classify_directory_contents(directory_path):
    """
    List and classify the contents of a given directory into folders and files.
    """
    logger.debug(f"Listing contents of {directory_path}")
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            print(f"{item} -> Folder")
        elif os.path.isfile(item_path):
            print(f"{item} -> File")
        else:
            print(f"{item} -> Unknown")

# Additional utility functions can be added here


