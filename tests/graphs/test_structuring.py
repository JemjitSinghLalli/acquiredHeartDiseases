import networkx as nx
from networkx.algorithms import tree
import numpy as np
import pandas as pd

from utils.graphs.structuring import get_directed_edges


def test_get_directed_edges():
    """Tests get_directed_edges()"""
    test_data = pd.DataFrame(
        data={f"col{i}": np.random.normal(size=100) for i in range(20)}
    )
    test_mst = tree.maximum_spanning_tree(nx.from_pandas_adjacency(test_data.corr()))
    test_directed_edges = get_directed_edges(test_mst, "col0")

    assert isinstance(
        test_directed_edges, list
    ), "get_directed_edges() is not returning a list as expected"
