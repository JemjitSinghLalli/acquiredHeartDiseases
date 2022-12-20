"""Tests for data preprocessing functions"""

import pandas as pd
from pytest import raises

from utils.preprocessing.generic_preprocessing import (
    reduce_data_frame_to_categorical_columns,
    reduce_data_frame_to_numeric_columns,
    bin_numeric_data,
)


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


def test_bin_numeric_data():
    """This tests bin_numeric_data() by passing in numeric data and checking for correct binning"""
    test_data_only_numeric_columns = pd.DataFrame(
        data={
            "col1": range(100),
            "col2": range(0, 1000, 10),
            "col3": range(0, 10000, 100),
            "col4": [1, 2] * 50,
        }
    )
    test_data_mixed_columns = pd.DataFrame(
        data={"col1": [5, 4, 3, 2, 1], "col2": ["hi", "this", "is", "a", "test"],}
    )

    test_bins = 3

    with raises(ValueError):
        bin_numeric_data(test_data_mixed_columns, test_bins)

    test_binned_data = bin_numeric_data(test_data_only_numeric_columns, test_bins)
    assert isinstance(
        test_binned_data, pd.DataFrame
    ), "bin_numeric_data() is not returning a pandas DataFrame as expected"

    for column in test_binned_data.columns:
        assert (
            test_binned_data[column].nunique() <= test_bins
        ), "bin_numeric_data() error, number of bins greater than expected"

        assert column.endswith(
            f"_{test_bins}bin"
        ), "expected columns were not found in dataframe output by bin_numeric_data()"


def test_reduce_data_frame_to_categorical_columns():
    """This tests reduce_data_frame_to_categorical_columns() by passing in different dataframes and checking that it
    returns the ones we passed in but only with numeric columns"""
    test_data_mixed_columns = pd.DataFrame(
        data={"col1": range(100), "col2": ["hi", "this", "is", "a", "test"] * 20,}
    )
    test_data_only_numeric_columns = pd.DataFrame(
        data={"col1": range(100), "col2": range(100), }
    )
    test_data_only_string_columns = pd.DataFrame(
        data={
            "col1": ["5", "4", "3", "2", "1"] * 20,
            "col2": ["1", "2", "3", "4", "5"] * 20,
        }
    )

    test_result_mixed = reduce_data_frame_to_categorical_columns(
        test_data_mixed_columns, cols_to_exclude=[]
    )
    test_result_numeric = reduce_data_frame_to_categorical_columns(
        test_data_only_numeric_columns, cols_to_exclude=[]
    )
    test_result_categorical = reduce_data_frame_to_categorical_columns(
        test_data_only_string_columns, cols_to_exclude=[]
    )
    for dataframe in [
        test_result_mixed,
        test_result_numeric,
        test_result_categorical,
    ]:
        assert isinstance(
            dataframe, pd.DataFrame
        ), "reduce_data_frame_to_categorical_columns() did not return pd.DataFrame"

    assert list(test_result_mixed.columns) == [
        "col2"
    ], "reduce_data_frame_to_categorical_columns() did not return expected columns for mixed type columns dataset"
    assert (
        list(test_result_numeric.columns) == []
    ), "reduce_data_frame_to_numeric_columns() did not return expected columns for numeric columns only dataset"
    assert list(test_result_categorical.columns) == [
        "col1",
        "col2",
    ], "reduce_data_frame_to_numeric_columns() did not return expected columns for mixed type columns dataset"
