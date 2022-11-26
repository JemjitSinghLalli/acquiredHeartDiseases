"""
This module is for functions that import data
"""
import os

import pandas as pd

path_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def import_csv_data(file_location: str = "data/data.csv") -> pd.DataFrame:
    """Reads out a pd.DataFrame from a .csv file.

    Args:
        file_location (str, optional): The .csv file location. Defaults to "data/data.csv".

    Returns:
        pd.DataFrame: Data ready for further processing.
    """
    return pd.read_csv(file_location)
