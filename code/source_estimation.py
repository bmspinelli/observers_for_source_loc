#Brunella Marta Spinelli (brunella.spinelli@epfl.ch)
"""This file contains some functions needed to estimate (via maximum
likelihood) the source of a SI epidemic process (with Gaussian or truncated
Gaussian edge delays).

The important function is 
    s_est, likelihood = ml_estimate(graph, obs_time, sigma, is_tree, paths,
    path_lengths, max_dist)

where s_est is the list of nodes having maximum a posteriori likelihood and
likelihood is a dictionary containing the a posteriori likelihood of every
node.

"""
import math
import networkx as nx
import numpy as np
import source_est_tools as tl

import scipy.stats as st
from scipy.misc import logsumexp

def ml_estimate(graph, obs_time, sigma, is_tree, paths, path_lengths,
        max_dist=np.inf):
    """Returns estimated source from graph and partial observation of the
    process.

    - graph is a networkx graph
    - obs_time is a dictionary containing the observervations: observer -->
      time
    
    Output: 
    - list of nodes having maximum a posteriori likelihood 
    - dictionary: node -> a posteriori likelihood
    
    """

    obs = obs_time.keys()  #observers.
    nodes = paths[obs[0]].keys()
    min_time = min(obs_time.values())
    o1 = filter(lambda x: obs_time[x]==min_time, obs_time.keys())[0]
    
    #initialize loglikelihood
    loglikelihood = {n: -np.inf for n in nodes}#node -> loglikelihood.
    

    #STEP 1: if only one observation...
    if len(obs) == 1:
        v = nodes
        v.remove(obs[0])
        # Same loglikelihood for every node.
        for candidate in v:
            loglikelihood[candidate] = - np.log(len(v))
        return v, posterior_from_logLH(loglikelihood)    


    #STEP 2: if sigma = 0
    if sigma == 0:
        classes, lens = tl.classes(path_lengths, obs)
        obs_D = np.zeros((len(obs)-1, 1)) 
        for l in range(1, len(obs)):
            obs_D[l-1] = obs_time[obs[l]] - obs_time[obs[0]]
        for c in classes:
            mu_s = tl.mu_vector_s(path_lengths, c[0], obs) 
            if max(abs(mu_s - obs_D)) < 10**(-6):
                equiv_nodes = c
        for n in equiv_nodes:
            loglikelihood[n] = -np.log(len(equiv_nodes))
        return equiv_nodes, posterior_from_logLH(loglikelihood)
    
    
    #STEP 2a: if the selected subgraph is a tree...
    if is_tree:
        #covariance matrix
        cov_d = tl.cov_mat(graph, graph, paths, obs) 
        for s in nodes: #loglikelihood for every node
            mu_s = tl.mu_vector_s(path_lengths, s, obs) #mu vector
            loglikelihood[s] = logLH_source_tree(mu_s, cov_d, obs, obs_time) 

    #STEP 2b: if it is not a tree and we have more than 1 observation...
    else:
        candidates = 0
        classes, _ = tl.classes(path_lengths, obs)
        for c in classes:
            tmp_lkl = []   
            for s in c:
                if path_lengths[o1][s] < max_dist:
                    tree_s = likelihood_tree(paths, s, obs) #BFStree
                    #covariance matrix
                    cov_d_s = tl.cov_mat(tree_s, graph, paths, obs)
                    mu_s = tl.mu_vector_s(path_lengths, s, obs) #mu vector
                    tmp_lkl.append(logLH_source_tree(mu_s, cov_d_s, obs, obs_time))
                    candidates += 1
            if len(tmp_lkl)>0:
                for s in c:
                    loglikelihood[s] = np.mean(tmp_lkl)

    #STEP 3: find the nodes with maximum loglikelihood and return the nodes
    #with maximum a posteriori likelihood
    posterior = posterior_from_logLH(loglikelihood)
    max_lkl = max(posterior.values()) #maximum of posterior
    arg_max = list()
    for src, value in posterior.items(): #find nodes with maximum loglikel.
        if np.isclose(value, max_lkl, atol= 1e-08):
            arg_max.append(src)
    s_est = arg_max 
    return s_est, posterior


def posterior_from_logLH(loglikelihood):
    """Returns a dictionary: node -> posterior probability
    
    """
    bias = logsumexp(loglikelihood.values())
    return dict((key, np.exp(value - bias))
            for key, value in loglikelihood.iteritems())


def logLH_source_tree(mu_s, cov_d, obs, obs_time):
    """ Returns loglikelihood of node 's' being the source

    - obs_time is a dictionary containing the observervations: observer --> time
    - obs is the ordered list of observers, i.e. obs[0] is the reference
    - mu_s is the mean vector of Gaussian delays when s is the source
    - cov_d the covariance matrix for the tree

    """
    assert len(obs) > 1
    
    obs_d = np.zeros((len(obs)-1, 1))
    for l in range(1, len(obs)):
        obs_d[l-1] = obs_time[obs[l]] - obs_time[obs[0]] # observer deltas
    exponent =  - ((obs_d - mu_s).T.dot(np.linalg.inv(cov_d)).dot(obs_d -
            mu_s))
    denom = math.sqrt(((2*math.pi)**len(obs_d))*np.linalg.det(cov_d))
    return (exponent - np.log(denom))[0,0]


def likelihood_tree_no_dependancies(paths, s, obs):
    """Compute the BFS tree from the source to the observers

    Dependent observers are not considered
    The relevant observers are returned in the list O_to_keep.
    ATTENTION: this function creates a problem because in some cases there is
    only one observers kept. Maybe this could be solved by randomly adding one 
    observer.
    """
    tree = nx.Graph()
    obs_to_keep = list()
    for o in obs:
        p = paths[o][s]
        #if the source is an observer the intersection is always larger than 1
        if len(set(p[:-1]).intersection(set(obs))) == 1:
            tree.add_edges_from(zip(p[0:-1], p[1:]))
            obs_to_keep.append(o)
    return tree, obs_to_keep


def likelihood_tree(paths, s, obs):
    tree = nx.Graph()
    for o in obs:
        p = paths[o][s]
        tree.add_edges_from(zip(p[0:-1], p[1:]))
    return tree


def preprocess(edges, obs, sigma, weight=None, distr='truncated'):
    """This function is intended to do some preprocessing of the graph before
    running simulations or source localisation on it. 
    
    It takes as INPUT
    - an edgelist
    - a list of observers
    - a value for the standard deviation \sigma
    - a dictionary of weights per edge

    The OUTPUT is
    - a networkx graph with weight and var attached to each edge
    - a boolean variable is_tree 
    - a dictionary of paths from observers to all other nodes
    - a dictionary of paths lengths from observers to all other nodes    
    
    """

    if distr == 'exp':
        #dummy value for sigma, in this case only edge weights matter
        assert sigma == -1

    #create a graph
    graph = nx.Graph()
    graph.add_edges_from(edges)

    
    #decide if it is tree
    is_tree = tl.is_tree(graph) 

    #prepare variance weights
    if weight == None:
        weight={(u,v): 1.0 for (u,v) in edges}
    
    #store variances values 
    var = {}
    for w in weight.values():
        if sigma == 0:
            var[w] = 0
        elif distr == 'truncated':
            var[w] = float(st.truncnorm.stats(-1.0/(2*sigma), 
                    1.0/(2*sigma), loc=w, scale=sigma*w, moments='v'))

    #variances = {u: {} for u in graph.nodes()}
    for (u, v) in edges:
        graph[u][v]['weight'] = weight[(u,v)]
        if sigma == 0:
            graph[u][v]['var'] = 0
        elif distr == 'truncated':
            graph[u][v]['var'] = var[weight[(u,v)]]
        elif distr == 'gaussian':
            graph[u][v]['var'] = (sigma*weight[(u,v)])**2
        elif distr == 'exp':
            #for the exponential rv st dev = mean
            graph[u][v]['var'] = weight[(u,v)]**2

    #I compute paths from obs to all other nodes and their lengths
    path_lengths = {}
    paths = {}
    for o in obs:
        path_lengths[o], paths[o] = nx.single_source_dijkstra(graph, o)
        #the attribute 'weight' is taken into account by default
    return graph, is_tree, paths, path_lengths
