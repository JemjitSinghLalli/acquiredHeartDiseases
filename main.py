"""
This is the basic runner for structuring graphs and performing ML over graphs
"""

from utils.load.data_importing import import_csv_data
from utils.preprocessing.generic_preprocessing import reduce_data_frame_to_numeric_columns
from utils.preprocessing.bespoke_preprocessing import convert_columns_to_correct_types


df = import_csv_data("data/heartDisease.csv")
df = convert_columns_to_correct_types(df)

numeric_df = reduce_data_frame_to_numeric_columns(df)

