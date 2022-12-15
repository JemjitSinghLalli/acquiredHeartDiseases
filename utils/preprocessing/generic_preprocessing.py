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
