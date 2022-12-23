"""
A module for the generation of probabiltiy distributions
"""
from itertools import product
from typing import Dict, List

import pandas as pd
from pomegranate import State, DiscreteDistribution, ConditionalProbabilityTable


def get_pd(series: pd.Series, unique_value_limit: int = 15) -> pd.DataFrame:
    """This will get a discrete distribution over the values of the `series`.

    Args:
        series (pd.Series): The series to get the probability distribution of.
        unique_value_limit (int): Will reject the distribution if there are too many unique values, defaults to 10.

    Returns: The probability distribution as a pd.DataFrame.

    """
    assert series.nunique() < unique_value_limit, (
        f"number of unique values in `series` must be less than `unique_value_limit ({unique_value_limit}) for get_pd()"
        " to work"
    )
    series = series.copy()
    name = series.name
    return (
        (series.value_counts() / series.shape[0])
        .sort_index()
        .reset_index()
        .astype({"index": str})
        .rename(columns={"index": name, name: "probability"})
    )


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
    for broadcast_node in broadcast_set:
        probability_distribution = get_pd(data[broadcast_node])
        pom_distribution = DiscreteDistribution(
            {
                category: probability
                for category, probability in probability_distribution.values
            }
        )
        state_dict[broadcast_node] = State(pom_distribution, name=broadcast_node)
        distribution_dict[broadcast_node] = pom_distribution

    from_nodes = broadcast_set.copy()
