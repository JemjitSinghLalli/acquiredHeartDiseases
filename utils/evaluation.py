"""This module is for "querying" models, which is functions for asking questions of models, such as 'what would happen
if this variable was set to the value X? Or what is likely to happen for car Y?'"""

from pomegranate import BayesianNetwork


def query_bayesian_network(model: BayesianNetwork, query: dict) -> dict:
    """This will take an existing `BayesianNetwork` and will run some queries over the data and return the outputs. The
    output is the conditional joint probability distribution of the nodes in the network not held in evidence. Where
    "holding in evidence" means passing in a value to the network for a node.

    Args:
        model: A trained pomegranate `BayesianNetwork`.
        state_name_order: The order the states were added to the network
        query: The query, which will be in the form of a dictionary, where the keys are the names of the nodes/state and
            the value is the value you want to set that state to in the network.

    Returns: The joint conditional probability distribution of the nodes not held in evidence, given the state of the
        nodes held in evidence.

    """
    state_name_order = [state.name for state in model.states]
    results = model.predict_proba(
        [[query[state_name] for state_name in state_name_order]]
    )[0]
    out_dict = dict()
    for i in range(len(results)):
        if query[state_name_order[i]] is None:
            out_dict[state_name_order[i]] = results[i].parameters[0]
    return out_dict
