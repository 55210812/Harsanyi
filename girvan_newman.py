"""Improved Girvan-Newman community detection algorithm with modularity-based stopping."""
import networkx as nx
from networkx.algorithms.community import modularity
from itertools import combinations

def girvan_newman(G, most_valuable_edge=None, weight="weight", threshold=0.1, target_num_communities=2):
    """Finds communities in a graph using the Girvan–Newman method with modularity stopping.

    Parameters
    ----------
    G : NetworkX graph
    most_valuable_edge : function, optional
        Function that takes a graph as input and outputs an edge. If None, uses modularity gain.
    weight : string or None, optional (default="weight")
        The edge attribute that holds the numerical value used as a weight.
    threshold : float, optional (default=0.0000001)
        Modularity gain threshold for stopping the algorithm.
    target_num_communities : int, optional
        Target number of communities to stop at. If None, uses modularity threshold.

    Returns
    -------
    list
        A list of sets (partition of `G`). Each set represents one community and contains
        all the nodes that constitute it.
    """
    if G.number_of_edges() == 0:
        return list(nx.connected_components(G))

    if most_valuable_edge is None:
        def most_valuable_edge(G,weight):
            """Returns the edge whose removal would result in the highest modularity gain."""
            current_mod = modularity(G, nx.connected_components(G), weight=weight)
            max_gain = -1
            best_edge = None
            
            # Evaluate each edge
            for edge in G.edges():
                H = G.copy()
                H.remove_edge(*edge)
                new_communities = tuple(nx.connected_components(H))
                new_mod = modularity(G, new_communities, weight=weight)
                gain = new_mod - current_mod
                if gain > max_gain:
                    max_gain = gain
                    best_edge = edge
            
            return best_edge if best_edge is not None else next(iter(G.edges()))

    g = G.copy().to_undirected()
    g.remove_edges_from(nx.selfloop_edges(g))
    
    # Initial communities and modularity
    communities = tuple(nx.connected_components(g))
    current_mod = modularity(G, communities, weight=weight)
    
    while g.number_of_edges() > 0:
        # Get next level of communities
        new_communities = _without_most_central_edges(g, most_valuable_edge,weight)
        
        # Calculate new modularity
        new_mod = modularity(G, new_communities, weight=weight)
        
        # Check stopping conditions
        if (target_num_communities is not None and len(communities) >= target_num_communities) :
            return list(communities)
            
        current_mod = new_mod
        communities = new_communities
    
    return list(communities)

def _without_most_central_edges(G, most_valuable_edge, weight="weight"):
    """Returns the connected components after removing the most valuable edge.
    
    Modified from networkx implementation to work with our modularity-based approach.
    """
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G,weight=weight)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    
    return new_components