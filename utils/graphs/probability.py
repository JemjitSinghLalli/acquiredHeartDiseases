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


def get_conditional_pd(
    data: pd.DataFrame, target: str, independent_variables: List[str]
) -> pd.DataFrame:
    """Gets conditional probability ovr the states of the target given the states of the independent variables.

    Args:
        data (pd.DataFrame): The data containing the `target` and `independent_variables`.
        target (pd.Series): The target series.
        independent_variables (List[str]): The independent variables.

    Returns: The conditional distribution of the `target` given the `independent_variables` as a pd.DataFrame.

    """
    data = data.copy()
    all_variables = independent_variables + [target]
    grouped_df = (
        data.groupby(all_variables)
        .count()
        .iloc[:, 0]
        .rename("count")
        .reset_index(drop=False)
        .set_index(independent_variables)
        .join(data.groupby(independent_variables).count().iloc[:, 0].rename("total"))
    )
    grouped_df["conditional_probability"] = grouped_df["count"] / grouped_df["total"]
    grouped_df = grouped_df.reset_index().set_index(all_variables)
    filler_df = pd.DataFrame(
        data=list(product(*(sorted(data[v].unique()) for v in all_variables))),
        columns=all_variables,
    ).set_index(all_variables)
    filler_df["count"] = filler_df["total"] = filler_df["conditional_probability"] = 0
    filler_df.update(grouped_df)
    return filler_df.reset_index()


def convert_cpdt_to_pomegranate_state(
    conditional_probability_distribution: pd.DataFrame,
    target: str,
    independent_variables: List[str],
    distribution_dict: dict,
) -> ConditionalProbabilityTable:

    """Takes a conditional probability table in the format output by get_conditional_pd() and turns it in to a
    corresponding `pomegranate.State`.

    Args:
        conditional_probability_distribution (pd.DataFrame): The conditional probability table in the format output by
            get_conditional_pd().
        target (str): The target of the probability distribution, this is the child node in the network.
        independent_variables (List[str]): The independent variables in the probability distribution, these are the
            parent nodes in the network.
        distribution_dict (dict): This associates nodes to their distributions and needs to be updated for future
            construction of the `BayesianNetwork`.

    Returns: A pomegranate `ConditionalProbabilityTable` corresponding to the input
        `conditional_probability_distribution`

    """

    conditional_probability_distribution = conditional_probability_distribution.astype(
        {x: str for x in independent_variables + [target]}
    )

    cpd = ConditionalProbabilityTable(
        [
            vals
            for vals in zip(
                *[
                    conditional_probability_distribution[var].to_list()
                    for var in independent_variables
                    + [target, "conditional_probability"]
                ]
            )
        ],
        [distribution_dict[key] for key in independent_variables],
    )

    distribution_dict[target] = cpd

    return State(cpd, name=target), distribution_dict


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

    while len(state_dict.keys()) < len(from_set.union(to_set)):
        nodes = directed_edge_df[directed_edge_df["from"].isin(from_nodes)][
            "to"
        ].unique()
        from_nodes = set()
        for node in nodes:
            nodes_edges = directed_edge_df[directed_edge_df["to"] == node]
            if all([x in state_dict.keys() for x in nodes_edges["from"]]):
                from_nodes.add(node)
                independent_variables = list(nodes_edges["from"].values)
                conditional_probability_distribution = get_conditional_pd(
                    data, node, independent_variables
                )
                state_dict[node], distribution_dict = convert_cpdt_to_pomegranate_state(
                    conditional_probability_distribution,
                    node,
                    independent_variables,
                    distribution_dict,
                )

    return state_dict
