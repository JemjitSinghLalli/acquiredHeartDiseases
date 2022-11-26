"""Tests data loading capabilities for the repository."""
import os

import pandas as pd

from utils.load.data_importing import import_csv_data

path_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_import_csv_data(tmp_path: str):
    """Ensures we can read .csv data into pd.DataFrame format where required.

    Args:
        tmp_path (str): Pytest inbuilt temporary data path.

    """
    pd.DataFrame(
        data={"col1": [5, 4, 3, 2, 1], "col2": ["hi", "this", "is", "a", "test"],}
    ).to_csv(f"{tmp_path}/test_import_csv_data.csv", index=False)

    test_frame = import_csv_data(f"{tmp_path}/test_import_csv_data.csv")
    assert isinstance(
        test_frame, pd.DataFrame
    ), "import_csv_data did not return pd.DataFrame"
    assert list(test_frame.columns) == [
        "col1",
        "col2",
    ], "import_csv_data returned unexpected columns"
