"""
A module for the generation of probabiltiy distributions
"""
from itertools import product
from typing import Dict, List

import pandas as pd
from pomegranate import State, DiscreteDistribution, ConditionalProbabilityTable


def get_pomegranate_states_from_directed_edges(
    data: pd.DataFrame, directed_edge_list: List[List[str]]
) -> Dict[str, State]:
    """This will take in `data`, the list of directed edges, and return a dictionary pointing each node, denoted by the
    name of the node, which corresponds to the name of the column in the `data`.

    Args:
        data (pd.DataFrame): The dataframe containing the columns corresponding to our nodes in our graph.
        directed_edge_list: The directed edges of our graph, a list of lists with 2 elements, pointing an edge from left
            to right.

    Returns: A dictionary that links each node's name to its state.

    """

    directed_edge_df = pd.DataFrame(directed_edge_list, columns=["from", "to"])
    from_set = set(directed_edge_df["from"])
    to_set = set(directed_edge_df["to"])
    broadcast_set = from_set.difference(to_set)
    state_dict = dict()
    distribution_dict = dict()
