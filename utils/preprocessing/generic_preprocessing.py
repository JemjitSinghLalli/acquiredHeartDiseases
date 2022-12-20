from typing import List

import pandas as pd


def reduce_data_frame_to_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and returns the same dataframe with only the columns that are numeric

    Args:
        df (pd.DataFrame): The dataframe to reduce to numeric columns

    Returns:
        pd.DataFrame: `data` with only numeric columns
    """
    numeric_columns = {
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    }
    return df.select_dtypes(include=numeric_columns)


def bin_numeric_data(numeric_data: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Get numeric columns and distribute them in to `bins` bins.

    Args:
        numeric_data (pd.DataFrame): The data to bin.
        bins (int): The number of bins to use.

    Returns: `numeric_data` with all columns binned.

    """
    numeric_columns = {
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    }
    if not all(
        [
            column_type.name in numeric_columns
            for column_type in numeric_data.dtypes.values
        ]
    ):
        raise ValueError(
            "All columns in `numeric_data` passed in to bin_numeric_data() must be numeric"
        )

    numeric_data = numeric_data.copy()

    binned_data = pd.DataFrame()
    for column in numeric_data.columns:
        binned_data[f"{column}_{bins}bin"] = pd.cut(
            numeric_data[column], bins, labels=False
        )

    return binned_data


def reduce_data_frame_to_categorical_columns(
    data: pd.DataFrame, cols_to_exclude: List[str], unique_value_limit: int = 15
) -> pd.DataFrame:
    """Takes a dataframe and returns the same dataframe with only the columns that are categorical.

    Args:
        data (pd.DataFrame): The dataframe to reduce to categorical columns
        cols_to_exclude (List[str]): The columns we do not want returned even if they are within the
        `unique_value_limit` unique_value_limit (int): The maximum number of unique values we let a categorical column
        have for inclusion.
        unique_value_limit: Maximum number of categories in categorical variables.

    Returns:
        pd.DataFrame: `data` with only categorical columns
    """
    categorical_columns = []
    for col in data.columns:
        if (1 < data[col].nunique() < unique_value_limit) and (
            col not in cols_to_exclude
        ):
            categorical_columns.append(col)
    categorical_data = data[categorical_columns]
    return categorical_data.apply(lambda x: x.astype("category").cat.codes)
