# Observer Placement for Source Localization

## Background

When an epidemic spreads in a network, a key question we want to answer is the one about the identity of its *source*, i.e., the node that started the epidemic. 

If we know the time at which various nodes were infected, we can attempt to use this information in order to identify the source. 
 
However, maintaining *observer* nodes that can provide their infection time is costly, and, in realistic scenarios, we have a *budget* *k* on the number of observer nodes we can maintain. 

Moreover, some nodes are more informative than others due to their location in the network. 

## Results

*Which nodes should we select as observers in order to maximize the probability that we can accurately identify the source?*

Inspired by the simple setting in which the node-to-node delays in the transmission of the epidemic are deterministic, we developed a principled approach for addressing the problem even when transmission delays are random.[1 ]

The optimal observer-placement differs depending on the *variance* of the transmission delays and propose approaches in both low- and high-variance settings.

## Datasets and materials

This repository contains the datasets and the Python implementation used for the experimental results presented in [1].


[1]: B.Spinelli, L.E.Celis and P.Thiran, *Observer placement for source localization: the effect of budgets and transmission variance*, Allerton Conf. on Communication, Control & Computing, 2017.
