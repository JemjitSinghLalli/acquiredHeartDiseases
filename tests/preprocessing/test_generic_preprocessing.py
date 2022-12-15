"""Tests for data preprocessing functions"""

import pandas as pd

from utils.preprocessing.generic_preprocessing import reduce_data_frame_to_numeric_columns


def test_reduce_data_frame_to_numeric_columns():
    """This tests reduce_data_frame_to_numeric_columns() by passing in different dataframes and checking that it returns
    the ones we passed in but only with numeric columns"""
    test_data_mixed_columns = pd.DataFrame(
        data={"col1": [5, 4, 3, 2, 1], "col2": ["hi", "this", "is", "a", "test"],}
    )
    test_data_only_numeric_columns = pd.DataFrame(
        data={"col1": [5, 4, 3, 2, 1], "col2": [1, 2, 3, 4, 5],}
    )
    test_data_only_string_columns = pd.DataFrame(
        data={"col1": ["5", "4", "3", "2", "1"], "col2": ["1", "2", "3", "4", "5"],}
    )

    test_result_mixed = reduce_data_frame_to_numeric_columns(test_data_mixed_columns)
    test_result_numeric = reduce_data_frame_to_numeric_columns(
        test_data_only_numeric_columns
    )
    test_result_strings = reduce_data_frame_to_numeric_columns(
        test_data_only_string_columns
    )
    for dataframe in [
        test_result_mixed,
        test_result_numeric,
        test_result_strings,
    ]:
        assert isinstance(
            dataframe, pd.DataFrame
        ), "reduce_data_frame_to_numeric_columns() did not return pd.DataFrame"
    assert list(test_result_mixed.columns) == [
        "col1",
    ], "reduce_data_frame_to_numeric_columns() did not return expected columns for mixed type columns dataset"
    assert list(test_result_numeric.columns) == [
        "col1",
        "col2",
    ], "reduce_data_frame_to_numeric_columns() did not return expected columns for numeric columns only dataset"
    assert (
        list(test_result_strings.columns) == []
    ), "reduce_data_frame_to_numeric_columns() did not return expected columns for mixed type columns dataset"
