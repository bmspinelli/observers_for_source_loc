"""This module contains several utility functions"""

import collections
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random



def dot(u, v):
    return sum(p*q for p,q in zip(u, v))

def mean(u, v):
    return [(p+q)/2.0 for p,q in zip(u, v)]

def norm2(u):
    return np.sqrt(sum(p**2 for p in u))

def mysum(u, v):
    return [p+q for p,q in zip(u, v)]

def diff(u, v):
    return [p-q for p,q in zip(u, v)]

def dist2(u, v):
    return norm2(diff(u,v))

def array_to_list(x):
    return [list(u).pop() for u in x]

# ----------------------------   GRAPH STRUCTURE


def find_leaves(graph): 
    """Find the leaves, i.e,  for a resolvable component the observers
    If the graph has only a node that node is considered a leaf.

    """
    leaves = list()
    if len(graph) == 1:
        return graph.nodes()
    for n in graph.nodes():
        if graph.degree(n) == 1:
            leaves.extend([n])
    return leaves


def degree_distr(graph):
    """Return the degree distribution of graph
    
    """
    d = list()
    for n in graph.nodes():
        d.append(graph.degree(n))
    return d


def distance_matrix(graph, nodes):
    """Returns distance matrix for ordered list of nodes in graph 

    """
    D = np.zeros([len(nodes), len(nodes)])
    for i, j in itertools.combinations(range(len(nodes)), 2):
         D[i][j] = nx.shortest_path_length(graph, nodes[i], nodes[j])
         D[j][i] = D[i][j]
    return D


def node_set_distance(graph, node, other_nodes):
    """Returns distance from a node to a set of nodes in graph 

    """
    m = np.inf
    for o in other_nodes:
        dist_o = nx.shortest_path_length(graph, node, o)
        if dist_o < m:
            m = dist_o
    return m


def is_tree(graph):
    """Check if a graph is a connected tree

    """
    return ((nx.number_of_nodes(graph) == nx.number_of_edges(graph) + 1) and
            nx.is_connected(graph))


def resolvable_nodes(tree, obs):
    """Compute all paths between pairs of observers in order to identify the
    resolvable nodes 

    """
    res_obs = set();
    paths = dict()
    for src, dst in itertools.combinations(obs, 2):
        nodes = set(nx.shortest_path(tree, source=src, target=dst))
        paths[(src, dst)] = nodes
        res_obs = res_obs.union(nodes)
    R = res_obs - set(obs); 
    return R


def resolvable_comps(tree, obs):
    """Identify all the resolvable components of T

    """
    list_res = list()
    res_nodes = resolvable_nodes(tree, obs)
    unres_nodes = set(tree.nodes()).difference(res_nodes) #unresov nodes + obs 
    #build a copy of tree
    tree_new = nx.Graph()
    tree_new.add_edges_from(tree.edges())
    tree_new.remove_nodes_from(obs) #remove obs to divide comps
    for component in nx.connected_component_subgraphs(tree_new):
        nodes = set(component.nodes())
        #select components that are not of kind L
        if len(nodes.intersection(res_nodes)) > 0: 
           #remove nodes that are not resolvable
           component.remove_nodes_from(unres_nodes.intersection(nodes))
           #attach observers
           for n in component.nodes():
               obs_neigh = set(tree.neighbors(n)).intersection(obs)
               for o in list(obs_neigh):
                   component.add_edge(n,o)
           #attach component to the list
           list_res.append(component)
    return list_res


def unresolvable_comps(tree, obs):
    """Identify all the unresolvable components of tree

    """
    list_u = list()
    res_nodes = list(resolvable_nodes(tree, obs))
    tree_new = tree.copy()
    tree_new.add_edges_from(tree.edges())
    tree_new.remove_nodes_from(res_nodes)
    tree_new.remove_nodes_from(obs)
    for component in nx.connected_component_subgraphs(tree_new):
        list_u.append(component)
    return list_u


def comps(tree, obs):
    comps = list() #components list
    tree_new = nx.Graph() #copy of tree
    tree_new.add_edges_from(tree.edges())
    tree_new.remove_nodes_from(obs) #disconnected tree

    #Add each observer to all its neighboring comps
    for component in nx.connected_component_subgraphs(tree_new):
        for node in component.nodes():
            for ne in tree.neighbors(node):
                if ne in obs:
                    component.add_edge(ne,node)
        comps.append(component)
    return comps

# ---------------------------- MU VECTORS and COV MATRIX  


def mu_vector_s(path_lengths, s, obs):
    """compute the mu vector for a candidate s

       obs is the ordered list of observers
    """
    v = list()
    for l in range(1, len(obs)):
        #the shortest path are contained in the bfs tree or at least have the
        #same length by definition of bfs tree
        v.append(path_lengths[obs[l]][s] - path_lengths[obs[0]][s])
    #Transform the list in a column array (needed for source estimation)
    mu_s = np.zeros((len(obs)-1, 1))
    mu_s[:, 0] = v
    return mu_s


def cov_mat(tree, graph, paths, obs):
    """Compute the covariance matrix of the observed delays.

    obs is the ordered set of observers. 

    """
    if not is_tree(tree):
        #if it is not a tree, the paths are not unique...
        raise ValueError("This function expects a tree!")

    #check if the shorrtest paths between obs[0] and the other 
    #observers are contained in the tree
    #and redefine them if this is not the case
    #NB no need to use the dijkstra networkx function because in a tree the
    #paths are unique
    bfs_tree_paths = {}
    for o in obs[1:]:
        if set(paths[obs[0]][o]) not in set(tree.nodes()):
            bfs_tree_paths[o] = nx.shortest_path(tree, obs[0], o)
        else:
            bfs_tree_paths[o] = paths[obs[0]][o]
    k = len(obs)
    cov = np.empty([k-1, k-1])
    for row in range(0, k-1):
        for col in range(0, k-1):
            if row == col:
                cov[row, col] = (total_variance(graph, 
                        bfs_tree_paths[obs[row+1]]))
            else:
                path_row = bfs_tree_paths[obs[row+1]]
                path_col = bfs_tree_paths[obs[col+1]] 
                common_nodes = filter(lambda x: x in path_col, path_row)
                p = nx.shortest_path(tree, common_nodes[0],common_nodes[-1])
                cov[row, col] = (total_variance(graph, p))
    return cov


def total_variance(graph, path):
    l = 0
    for i in range(len(path)-1):
        l += graph[path[i]][path[i+1]]['var']
    return l


# ---------------------------- Filtering diffusion data

def filter_diffusion_data(infected, obs, max_obs=np.inf):
    """Takes as input two dictionaries containing node-->infection time and
    node-->infector and filter only the items that correspond to observer nodes

    """
    obs_time = dict((k,v) for k,v in infected.iteritems() if k in obs)
    if max_obs < len(obs_time):
        max_time = sorted(obs_time.values())[max_obs]
        #obs_time = dict((k,v) for k,v in obs_time.iteritems() if (v < max_time))
        node_time = obs_time.items()
        random.shuffle(node_time)
        #print node_time
        new_obs_time = {}
        (n, t) = node_time.pop()
        while len(new_obs_time) < max_obs:
            if t <= max_time:
                new_obs_time[n] = t
            (n, t) = node_time.pop()
        return new_obs_time 
    else:
        return obs_time

# ---------------------------- Equivalence classes

def classes(paths, b):
    u = b[0]
    vector_to_n = collections.defaultdict(list)
    for n in paths[u].keys():
        vector_to_n[tuple(int((10**8)*(paths[v][n] - paths[u][n])) for v in b[1:])].append(n)
    classes = vector_to_n.values()
    return classes, [len(c) for c in classes]


def draw_graph(graph, pos=None, obs=None, source=None, est_source=None, 
        obs_col='g', source_line=2.0, est_col='c', labels=False, node_size=100, 
        plot=False, save=False, alpha=0.2, output='output.eps'):
    """Drawing helper function

    """

    plt.clf()
    plt.axis('off')
    if pos == None:
        pos = nx.get_node_attributes(graph,'pos')
        if len(pos) == 0:
            pos = nx.graphviz_layout(graph)
    colors = {}
    lines = {}
    for n in graph.nodes():
        colors[n] = 'w'
        lines[n] = 1.0
    if obs != None:
        for o in obs:
            colors[o] = obs_col
    if est_source != None:
        for n in est_source:
            colors[n] = est_col
    if source != None:
        lines[source] = source_line
    color = list(colors[n] for n in graph.nodes())
    line = list(lines[n] for n in graph.nodes())
    nx.draw_networkx_nodes(graph, pos, node_color=color, node_size=node_size,
            alpha=1, linewidths=line)
    nx.draw_networkx_edges(graph, pos, edge_color='b', alpha=alpha)
    if labels:
        nx.draw_networkx_labels(graph, pos, font_size=10) 
    if plot:
        fig = plt.gcf()
        fig.set_size_inches(18.5,10.5)
        plt.show()
    if save: 
        plt.savefig(output)
