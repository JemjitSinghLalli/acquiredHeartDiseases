from typing import List

import networkx as nx
import pandas as pd


def get_directed_edges(graph: nx.Graph, target: str) -> List[List[str]]:
    """This will order the pairs of nodes in the ego graph from left to right based on direction towards the target.

    Args:
        graph (nx.Graph): Any NetworkX graph
        target (str): Node in the network which we point edges to

    Returns: a list of nodes, in pairs, showing their direction from left to right
    """
    assert (
        target in graph.nodes
    ), "`target` must be in `graph` for get_directed_edges() to work!"
    edge_list = list(graph.edges())
    directed_edge_df = pd.DataFrame(edge_list, columns=["from", "to"])
    directed_edge_df = directed_edge_df.append(
        directed_edge_df.rename(columns={"from": "to", "to": "from"})
    )

    directed_edge_list = directed_edge_df[
        directed_edge_df["to"] == target
    ].values.tolist()
    new_node_set = {x[0] for x in directed_edge_list}
    discovered_nodes = {target}
    directed_edge_df = directed_edge_df[~(directed_edge_df == target).any(axis=1)]
    while len(directed_edge_list) < len(graph.edges):
        new_edge_list = directed_edge_df[
            directed_edge_df["to"].isin(new_node_set)
            & (~directed_edge_df["to"].isin(discovered_nodes))
        ].values.tolist()
        directed_edge_list += new_edge_list
        discovered_nodes = discovered_nodes.union(new_node_set)
        directed_edge_df = directed_edge_df[
            ~directed_edge_df.isin(new_node_set).any(axis=1)
        ]
        new_node_set = {x[0] for x in new_edge_list}

    return directed_edge_list
