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
