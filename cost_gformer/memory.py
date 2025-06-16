"""Memory buffers for CoST-GFormer.

The memory system is separated into three conceptual blocks:

- :class:`UltraShortTermAttention` (USTA): a very small buffer for the most
  recent tokens used directly by attention layers.
- :class:`ShortTermMemory` (STM): holds recent context for quick retrieval.
- :class:`LongTermMemory` (LTM): archives older information for long-range
  dependencies.

These classes are placeholders and only document the intended behavior.
"""

import numpy as np


class UltraShortTermAttention:
    """Represents the USTA buffer."""

    def __init__(self, size: int = 4):
        self.size = size

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(size={self.size})"


class ShortTermMemory:
    """Represents the STM buffer."""

    def __init__(self, size: int = 128):
        self.size = size

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(size={self.size})"


class LongTermMemory:
    """Lightweight memory storing historical node embeddings."""

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int,
        num_centroids: int = 8,
        rng: "np.random.Generator | None" = None,
    ) -> None:

        self.num_nodes = int(num_nodes)
        self.embed_dim = int(embed_dim)
        self.num_centroids = int(num_centroids)

        rng = np.random.default_rng() if rng is None else rng
        self.rng = rng

        self.centroids = np.zeros(
            (self.num_nodes, self.num_centroids, self.embed_dim), dtype=np.float32
        )
        # Gating projection for fuse()
        self.W_g = (
            rng.standard_normal((2 * self.embed_dim, self.embed_dim), dtype=np.float32)
            / np.sqrt(2 * self.embed_dim)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(x: "np.ndarray") -> "np.ndarray":
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    @staticmethod
    def _sigmoid(x: "np.ndarray") -> "np.ndarray":
        return 1.0 / (1.0 + np.exp(-x))

    # ------------------------------------------------------------------
    def build(self, embeddings: "np.ndarray", iters: int = 10) -> None:
        """Cluster embeddings to initialise the centroids.

        Parameters
        ----------
        embeddings:
            Array of shape ``(steps, num_nodes, embed_dim)`` containing
            historical node embeddings.
        iters:
            Number of k-means iterations to perform.
        """


        if embeddings.ndim != 3:
            raise ValueError("embeddings must be (steps, nodes, dim)")

        steps, nodes, dim = embeddings.shape
        if nodes != self.num_nodes or dim != self.embed_dim:
            raise ValueError("embedding shape mismatch")

        for v in range(self.num_nodes):
            X = embeddings[:, v, :]
            n_samples = X.shape[0]
            if n_samples == 0:
                continue

            # Initialise centroids with random samples
            idx = self.rng.choice(n_samples, self.num_centroids, replace=n_samples < self.num_centroids)
            centers = X[idx].copy()

            for _ in range(max(1, iters)):
                # assign points to nearest centroid
                dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                assign = dists.argmin(axis=1)
                # update centroids
                for k in range(self.num_centroids):
                    mask = assign == k
                    if mask.any():
                        centers[k] = X[mask].mean(axis=0)

            self.centroids[v] = centers

    # ------------------------------------------------------------------
    def read(self, node: int, embedding: "np.ndarray") -> "np.ndarray":
        """Retrieve context vector for a node embedding."""

        cents = self.centroids[node]
        sims = embedding @ cents.T
        weights = self._softmax(sims)
        return weights @ cents

    def fuse(self, node: int, embedding: "np.ndarray") -> "np.ndarray":
        """Fuse current embedding with retrieved memory."""

        mem = self.read(node, embedding)
        gate = self._sigmoid(np.concatenate([embedding, mem]) @ self.W_g)
        return gate * embedding + (1.0 - gate) * mem

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return (
            f"{self.__class__.__name__}(nodes={self.num_nodes}, "
            f"dim={self.embed_dim}, centroids={self.num_centroids})"
        )
