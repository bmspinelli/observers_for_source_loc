#!/bin/src/python
from __future__ import print_function

import cPickle
import itertools
import networkx as nx
import sys


graph = sys.argv[1]
s = int(sys.argv[2])

with open(graph, 'r') as f:
    g = cPickle.load(f)

count = {i: 0 for i in g.nodes()}
for t in range(s):
    paths = list(nx.all_shortest_paths(g, s, t))
    concatenated_paths = list(itertools.chain(*paths))
    total = len(paths)
    for n in concatenated_paths:
        if n not in {s, t}:
            count[n] += 1.0/total
for n in range(len(g)):
    print(count[n], end=" ")
