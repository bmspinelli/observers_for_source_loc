# Generate hn-Obs 

import networkx as nx
import random

def place_observers(graph, budget, l):
    d = nx.all_pairs_dijkstra_path_length(g)
    paths = nx.all_pairs_dijkstra_path(g)

    assert budget > 1
    assert len(d) == len(paths)
    assert l <= max([max(d[v].values()) for v in d])

    best_value = 0
    for first in d.keys():
        obs, value, covered = _complete_placement(d, paths, [first], budget, l)
        if value > best_value:
            best_obs = obs
            best_value = value
            best_covered = covered
    return best_obs, best_value, best_covered


def _complete_placement(d, paths, selected, budget, l):
    
    assert len(selected)<budget

    if len(selected)==1:
        covered = set()
        value = 0
    else:
        raise ValueError("not ready to complete this set") 

    while len(selected)<budget and value < len(d.keys()):
       candidates = filter(lambda x: x not in selected, d.keys())
       random.shuffle(candidates)
       improvement = 0
       for c in candidates:
           #count how many nodes get covered when adding c
           count, newly_covered = __update(d, paths, l, covered, selected, c)
           if count > improvement:
               improvement = count
               best_covered = newly_covered
               best_candidate = c
       if improvement == 0:
           break
       else:
           selected.append(best_candidate)
           covered = covered.union(best_covered)
           value += improvement
    return selected, value, covered


def __update(d, paths, l, covered, selected, c):
    "If there are no nodes at distance smaller than l returns 0, []"
    newly_covered = set()
    for obs in selected:
        if d[c][obs]<= l and paths[c][obs]:
            for p in paths[c][obs]:
                newly_covered = newly_covered.union(set(p))
    newly_covered = newly_covered.difference(covered)
    return len(newly_covered), newly_covered
