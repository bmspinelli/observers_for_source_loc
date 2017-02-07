#!/usr/src/python

import cPickle
import networkx as nx
import numpy as np
import sys

import diffusion as diff
import source_estimation as se
import source_est_tools as tl

graph_path = sys.argv[1]
obs_path = sys.argv[2]
source = int(sys.argv[3])
sigma = float(sys.argv[4])
max_obs = int(sys.argv[5])
max_dist = int(sys.argv[6])
distr = sys.argv[7]

samples = 20

#read graph
with open(graph_path, 'r') as f:
    graph = cPickle.load(f)

edges = graph.edges()
        
#read observers
with open(obs_path, 'r') as f:
    obs = cPickle.load(f)[0]

#preprocess the graph
graph, is_tree, paths, path_lengths = se.preprocess(edges, obs, sigma,
        weight={(u,v): graph[u][v]['weight'] for (u,v) in graph.edges()},
        distr=distr)

for i in range(samples):
    #generate a diffusion
    infected = diff.diffusion(graph, sigma, source, distr=distr)

    #filter data
    obs_time = tl.filter_diffusion_data(infected, obs, max_obs=max_obs)

    #run the estimation
    s_est, posterior = se.ml_estimate(graph, obs_time, sigma, is_tree, paths,
        path_lengths, max_dist=max_dist)    

    if source in s_est:
        source_in_est = 1
    else: 
        source_in_est = 0

    #print result: avg distance of nodes that maximize lkl, success, number of
    #sources that maximize the lkl
    print sigma, source, np.mean([nx.shortest_path_length(graph, source,
            s) for s in s_est]), source_in_est, len(s_est)

    #for WEIGHTED distance use nx.dijkstra_path_length
