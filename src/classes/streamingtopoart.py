"""
This package provides TopoSaliencMap library.

This file provides the Streaming TopoART module.
"""

import os
from typing import Dict, List, Tuple, IO
from operator import itemgetter
import numpy as np
import networkx as nx
from pyvis.network import Network
import colorsys
import webcolors
from . base import *
from .. functions import *
import ipdb


__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2024"
__credits__ = ["Leonardo E B Da Silva","Donald Wunsch"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Raghu Yelugam"
__email__ = "ry222@mst.edu"
__status__ = "Development"
__date__ = "2024.02.08"



class StreamingTopoART(BaseART):
    """
    Reference: Tscherepanow, M., 2010. TopoART: A topology learning hierarc
    -hical ART network. In Artificial Neural Networks–ICANN 2010: 20th Inte
    -rnational Conference, Thessaloniki, Greece, September 15-18, 2010,
    Proceedings, Part III 20 (pp. 157-167). Springer Berlin Heidelberg.
    """


    def __init__(self,
                 vigilance: float,
                 alpha: float,
                 beta: float,
                 beta2: float,
                 phi: int,
                 tau: int) -> None:
        """
        Constructor for the TopoART algorithm.

        Arguments:
        :param vigilance: The vigilance of the algorithm.
        :param alpha: The alpha parameter of the algorithm.
        :param beta: The first winner learning rate.
        :param beta2: The second winner learning rate
        :param phi: The minimum time difference before deleting a prototype
        :param tau: The minimum number cycles to commence pruning
        """
        super().__init__(vigilance, alpha, beta)
        self.beta2: float = beta2
        self.phi: int = phi
        self.tau: float = tau
        self.cycle: int = 0
        self.clusters: List[List[str]] = []
        self.network = nx.Graph()
        
    def __repr__(self):
        doc = (f"Arguments: \n"
                f":param vigilance: The vigilance of the algorithm, {self.vigilance}. \n"
                f":param alpha: The alpha parameter of the algorithm, {self.alpha}. \n"
                f":param beta: The first winner learning rate, {self.beta}. \n"
                f":param beta2: The second winner learning rate, {self.beta2}. \n"
                f":param phi: The minimum time difference before deleting a prototype, {self.phi}. \n"
                f":param tau: The minimum number cycles to commence pruning, {self.tau}. \n")
        print(doc)

    def __add_prototype(self,
                        weights: np.ndarray,
                        tag: str,
                        birth: int) -> None:
        """
        Adds new prototype to the ART module and adds 
        a new node to the network
        
        Arguments:
        :param weights: weights of the prototype
        :param tag: tag to identify the prototype
        :param birth: birth time of the prototype
        """
        self.prototypes.append(prototype(weights,
                                         tag,
                                         birth))
        self.network.add_node(tag)
    
    def __prune(self) -> None:
        """
        Delete prototypes with set criterion
        """
        deletedtags: List[str] = []
        i = 0
        while i < len(self.prototypes):
            if (self.cycle - self.prototypes[i].lastactive) > self.phi:
                deletedtags.append(self.prototypes[i].tag)
                del self.prototypes[i]
                i -= 1
            i += 1
            
        self.network.remove_nodes_from(deletedtags)

    def learn(self,
              input: np.ndarray) -> None:
        """
        Implements a learn function
        
        Arguments:
        :param input: the input vector to be fet to the ART model
        """
        self.cycle += 1
        if len(self.prototypes) == 0:
            self.__add_prototype(input,
                                 f"p{self.cycle}",
                                 self.cycle)
        else:
            T = super().choice(input)
            M = super().match(input)
            while not all(val < 0.0 for val in T):
                IFW: int = T.index(max(T))
                if M[IFW] >= self.vigilance:
                    self.prototypes[IFW].weights = \
                    (1 - self.beta)*self.prototypes[IFW].weights \
                    + self.beta*np.minimum(input, self.prototypes[IFW].weights)
                    self.prototypes[IFW].counter += 1
                    T[IFW] = -1.0
                    tagFW = self.prototypes[IFW].tag
                    while not all(val2 < 0 for val2 in T):
                        ISW: int = T.index(max(T))
                        if M[ISW] >= self.vigilance:
                            self.prototypes[ISW].weights = \
                                (1 - self.beta)*self.prototypes[IFW].weights \
                                + self.beta*np.minimum(input, self.prototypes[IFW].weights)
                            tagSW = self.prototypes[ISW].tag
                            if (tagFW, tagSW) not in list(self.network.edges) or \
                            (tagSW, tagFW) not in list(self.network.edges):
                                self.network.add_edge(tagFW,tagSW)
                            break
                        else:
                            T[ISW] = -1.0
                    break
                else:
                    T[IFW] = -1.0
            if not any(val > 0 for val in T):
                self.__add_prototype(input,
                                     f"p{self.cycle}",
                                     self.cycle)
        #prune
        if self.cycle > self.tau:
            self.__prune()

    def get_graph(self) -> IO:
        """
        Generate a graph with connected components node sharing the same color
        
        Returns:
        Graph: returns a pyvis graph object
        """
        connected_components = list(nx.connected_components(self.network))
        n_colors = generate_colors(len(connected_components))
        gX = Network(notebook = True)
        gX.from_nx(self.network)
        for node in gX.nodes:
            T = True
            i = 0
            while T:
                if node["id"] in connected_components[i]:
                    T = False
                    node["color"] = n_colors[i]
                i += 1
        return gX