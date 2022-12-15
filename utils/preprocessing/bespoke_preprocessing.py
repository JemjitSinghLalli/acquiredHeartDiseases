import pandas as pd


def convert_columns_to_correct_types(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and returns the same dataframe with columns in their correct dtype.

    Args:
        df (pd.DataFrame): The dataframe to convert columns to correct type.

    Returns:
        pd.DataFrame: `data` with correct column types.
    """
    df["Sex"] = df["Sex"].astype("str")
    df["ExAng"] = df["ExAng"].astype("str")

    return df
