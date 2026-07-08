import numpy as np
import torch


class HandGraph:
    """26-node hand skeleton graph with spatial partition strategy."""

    NUM_NODES = 26
    CENTER_NODE = 0

    # Palm connects to each finger's metacarpal_prev
    PALM_EDGES = [(0, 1), (0, 6), (0, 11), (0, 16), (0, 21)]

    # Intra-finger edges (sequential joints)
    FINGER_EDGES = []
    for finger_idx in range(5):
        base = 1 + finger_idx * 5
        for j in range(4):
            FINGER_EDGES.append((base + j, base + j + 1))

    EDGES = PALM_EDGES + FINGER_EDGES  # 25 edges total

    def __init__(self):
        self.A = self._build_adjacency()

    def _get_hop_distance(self):
        """BFS from center node to compute distance of each node."""
        adj = np.zeros((self.NUM_NODES, self.NUM_NODES), dtype=int)
        for i, j in self.EDGES:
            adj[i, j] = 1
            adj[j, i] = 1

        dist = np.full(self.NUM_NODES, -1, dtype=int)
        dist[self.CENTER_NODE] = 0
        queue = [self.CENTER_NODE]
        while queue:
            node = queue.pop(0)
            for neighbor in range(self.NUM_NODES):
                if adj[node, neighbor] and dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        return dist

    def _normalize(self, A):
        """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.where(D > 0, np.power(D, -0.5), 0).astype(np.float32)
        D_mat = np.diag(D_inv_sqrt)
        return (D_mat @ A @ D_mat).astype(np.float32)

    def _build_adjacency(self):
        """Build 3-subset spatial partition adjacency matrices [3, 26, 26]."""
        dist = self._get_hop_distance()

        # Subset 0: self-loop (identity)
        A_self = np.eye(self.NUM_NODES, dtype=np.float32)

        # Subset 1: centripetal (neighbor closer to root)
        A_centripetal = np.zeros((self.NUM_NODES, self.NUM_NODES), dtype=np.float32)

        # Subset 2: centrifugal (neighbor farther from root)
        A_centrifugal = np.zeros((self.NUM_NODES, self.NUM_NODES), dtype=np.float32)

        for i, j in self.EDGES:
            if dist[j] < dist[i]:
                # j is closer to root -> centripetal for node i
                A_centripetal[i, j] = 1
                A_centrifugal[j, i] = 1
            elif dist[j] > dist[i]:
                # j is farther from root -> centrifugal for node i
                A_centrifugal[i, j] = 1
                A_centripetal[j, i] = 1
            else:
                # same distance: split into centripetal
                A_centripetal[i, j] = 1
                A_centripetal[j, i] = 1

        A_self = self._normalize(A_self)
        A_centripetal = self._normalize(A_centripetal + np.eye(self.NUM_NODES))
        A_centrifugal = self._normalize(A_centrifugal + np.eye(self.NUM_NODES))

        A = np.stack([A_self, A_centripetal, A_centrifugal], axis=0)
        return torch.from_numpy(A)


def get_adjacency():
    """Return adjacency tensor [3, 26, 26] for use in model."""
    graph = HandGraph()
    return graph.A
