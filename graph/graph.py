# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2022/4/3 10:54
import numpy as np


class Graph(object):
    """Sparse graph construction"""

    def __init__(self, num_node, num_k):
        self.num_node = num_node
        self.num_k = num_k

        self.get_binary_adj()
        self.normalize_adj()

    def get_binary_adj(self):
        self.adj = np.zeros((self.num_node, self.num_node))

        for i in range(self.num_node):
            for j in range(self.num_node):
                if (i - j < self.num_k and i - j >= 0) or (j - i < self.num_k and j - i >= 0):
                    self.adj[i, j] = 1
                else:
                    self.adj[i, j] = 0

    def normalize_adj(self):
        node_degrees = self.adj.sum(-1)
        degs_inv_sqrt = np.power(node_degrees, -0.5)
        norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
        self.norm_adj = (norm_degs_matrix @ self.adj @ norm_degs_matrix).astype(np.float32)



if __name__ == '__main__':
    graph = Graph(num_node=10, num_k=2)
    print(graph.adj)
    print(graph.norm_adj)