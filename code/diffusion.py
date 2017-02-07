#Brunella Marta Spinelli (brunella.spinelli@epfl.ch)
"""Simulate an SI epidemic

Edges delays are supposed to be truncated Gaussians with mean \mu and standard
deviation \sigma.

"""

import numpy.random as nprand
import scipy.stats as st

def diffusion(graph, sigma, source, distr='truncated', randseed=None):
    """Simulation of a PTV diffusion process.

    - graph: networkx graph
    - sigma: st.dev. of the edge delays
    - source: source node
    
    Output: dictionary mapping nodes to their infection time. Something like:
		{ node1: infection_time1, node2: infection_time2, ... }.

    """
    nprand.seed(randseed)  # Initialize the random seed.

    infected    = {source: 0}  # Infection time per node
    processing  = {source: 0}  # Infected nodes to process and their infection
                               # times
    
    while processing:
        node, time = sorted(processing.items(), key=lambda x: x[1], reverse=True).pop()
        for neighbour in graph.neighbors(node):
            try:
                weight = graph[node][neighbour]['weight']
            except KeyError:
                weight = 1.0
            infection_time = time + edge_delay(weight, sigma, distr)
            if neighbour not in infected or infected[neighbour] > infection_time:
                infected[neighbour] = infection_time
                processing[neighbour] = infection_time
        del processing[node]
    return infected


def edge_delay(mu, sigma, distr='truncated'):
    """If \sigma equals 0 nprand.normal() fails. This function handles this.

    """
    assert (distr == 'truncated') or (distr == 'gaussian') or (distr == 'exp')

    if distr == 'exp':
        #dummy value
        assert sigma == -1

    if sigma == 0:
        return mu
    else:
        if distr == 'truncated':
            a = (mu/2.0 - mu)/(mu*sigma)
            b = (3*mu/2.0 - mu)/(mu*sigma)
            return st.truncnorm.rvs(a, b, loc=mu, scale=mu*sigma)
        elif distr == 'gaussian':
            return max(nprand.normal(mu, mu*sigma), 0)
        elif distr == 'exp':
            #the variance of an exponential is determined by its mean
            return nprand.exponential(scale=mu)
