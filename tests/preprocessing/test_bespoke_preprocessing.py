"""Tests for data preprocessing functions"""

import pandas as pd
from pandas.api.types import is_string_dtype

from utils.preprocessing.bespoke_preprocessing import convert_columns_to_correct_types


def test_convert_columns_to_correct_types():
    """This tests convert_columns_to_correct_types() by passing in a dataframe and checking that it returns
    changed column types."""
    test_df = pd.DataFrame(data={"Sex": [1, 0, 0, 1, 1], "ExAng": [0, 1, 0, 0, 1],})
    test_df["Sex"] = pd.to_numeric(test_df["Sex"])
    test_df["ExAng"] = pd.to_numeric(test_df["ExAng"])

    correct_types_df = convert_columns_to_correct_types(test_df)

    assert isinstance(
        correct_types_df, pd.DataFrame
    ), "convert_columns_to_correct_types() did not return pd.DataFrame"
    assert is_string_dtype(
        correct_types_df.dtypes["Sex"]
    ), "column has not been converted to numeric."
