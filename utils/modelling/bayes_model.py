"""Functions for producing the Bayes Belief Network"""
from typing import Dict, List

from pomegranate import BayesianNetwork, State


def get_bayesian_network(
    state_dict: Dict[str, State], directed_edge_list: List[List[str]]
) -> BayesianNetwork:
    """This will take in a state dictionary which maps column/node names to pomegranate States which become nodes in the
    `pomegranate.BayesianNetwork`. This also requires the `directed_edge_list` so that the dependency structure of the
    graph is created correctly.

    `pomegranate.BayesianNetwork`: https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html
    Bayesian Networks, sometimes called 'decision networks', exist to help make decisions regarding observations (in our
    case cars) given input data. So in our case the point of making this network is to allow us to make good decisons
    about the treatment path and sales channel of the cars we process.

    Args:
        state_dict (Dict[str, State]): Maps node/column names to pomegranate states.
        directed_edge_list (List[List[str]]: A list of lists, where each sub-list is 2 nodes, on the left is the parent,
            on the right is the child, so the edge points left to right.

    Returns: a `pomegranate.BayesianNetwork` of the states in the `state_dict` given the dependency structure in the
        `directed_edge_list`.

    """
    model = BayesianNetwork("Bayes Net")
    state_name_order = []
    for state in state_dict.values():
        state_name_order.append(state.name)
        model.add_states(state)
    for from_node, to_node in directed_edge_list:
        model.add_edge(state_dict[from_node], state_dict[to_node])
    model.bake()
    return model, state_name_order
