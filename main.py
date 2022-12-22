"""
This is the basic runner for structuring graphs and performing ML over graphs
"""
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import tree
from sklearn.metrics import mutual_info_score

from utils.graphs.structuring import get_directed_edges
from utils.load.data_importing import import_csv_data
from utils.preprocessing.generic_preprocessing import (
    reduce_data_frame_to_numeric_columns,
    bin_numeric_data,
    reduce_data_frame_to_categorical_columns,
)
from utils.preprocessing.bespoke_preprocessing import convert_columns_to_correct_types


df = import_csv_data("data/heartDisease.csv")
df = convert_columns_to_correct_types(df)

numeric_df = reduce_data_frame_to_numeric_columns(df)
numeric_df = bin_numeric_data(numeric_df, 5)
categorical_df = reduce_data_frame_to_categorical_columns(df, list(numeric_df.columns))
training_df = numeric_df.join(categorical_df)
correlation_matrix = training_df.corr(method=mutual_info_score)
del categorical_df
del numeric_df
graph = tree.maximum_spanning_tree(nx.from_pandas_adjacency(correlation_matrix))
nx.draw(graph, with_labels=True)
plt.show()
directed_edge_list = get_directed_edges(graph, target="AHD")
