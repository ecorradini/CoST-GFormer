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
    """Cyclic buffer storing recent node embeddings."""

    def __init__(self, size: int = 128, num_nodes: int | None = None, embed_dim: int | None = None) -> None:
        self.size = int(size)
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.embed_dim = None if embed_dim is None else int(embed_dim)
        self._ptr = 0
        self._filled = 0
        self.buffer: np.ndarray | None = None

        if self.num_nodes is not None and self.embed_dim is not None:
            self._init_buffer()

    # ------------------------------------------------------------------
    def _init_buffer(self) -> None:
        self.buffer = np.zeros(
            (self.size, self.num_nodes, self.embed_dim), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def write(self, embeddings: "np.ndarray") -> None:
        """Store a new set of node embeddings."""

        if self.buffer is None:
            self.num_nodes, self.embed_dim = embeddings.shape
            self._init_buffer()
        if embeddings.shape != (self.num_nodes, self.embed_dim):
            raise ValueError("embedding shape mismatch")

        self.buffer[self._ptr] = embeddings
        self._ptr = (self._ptr + 1) % self.size
        self._filled = min(self._filled + 1, self.size)

    # ------------------------------------------------------------------
    def read(self, node: int) -> "np.ndarray":
        """Return the mean embedding for ``node`` across the buffer."""

        if self.buffer is None or self._filled == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)

        data = self.buffer[: self._filled, node]
        return data.mean(axis=0)

    def read_all(self) -> "np.ndarray":
        """Return mean embeddings for all nodes."""

        if self.buffer is None or self._filled == 0:
            return np.zeros((self.num_nodes, self.embed_dim), dtype=np.float32)

        data = self.buffer[: self._filled]
        return data.mean(axis=0)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(size={self.size})"


class LongTermMemory:
    """Lightweight memory storing historical node embeddings."""

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int,
        num_centroids: int = 8,
        buffer_size: int = 1024,
        rng: "np.random.Generator | None" = None,
    ) -> None:

        self.num_nodes = int(num_nodes)
        self.embed_dim = int(embed_dim)
        self.num_centroids = int(num_centroids)
        self.buffer_size = int(buffer_size)

        rng = np.random.default_rng() if rng is None else rng
        self.rng = rng

        self.buffer = np.zeros(
            (self.buffer_size, self.num_nodes, self.embed_dim), dtype=np.float32
        )
        self._ptr = 0
        self._filled = 0

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
    def write(self, embeddings: "np.ndarray") -> None:
        """Store new embeddings in the circular buffer."""

        if embeddings.shape != (self.num_nodes, self.embed_dim):
            raise ValueError("embedding shape mismatch")

        self.buffer[self._ptr] = embeddings
        self._ptr = (self._ptr + 1) % self.buffer_size
        self._filled = min(self._filled + 1, self.buffer_size)

    # ------------------------------------------------------------------
    def build(self, embeddings: "np.ndarray | None" = None, iters: int = 10) -> None:
        """Cluster embeddings to initialise the centroids.

        Parameters
        ----------
        embeddings:
            Optional array of shape ``(steps, num_nodes, embed_dim)`` containing
            historical node embeddings. If ``None``, the internal buffer is used.
        iters:
            Number of k-means iterations to perform.
        """

        if embeddings is None:
            if self._filled == 0:
                return
            embeddings = self.buffer[: self._filled]

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

    def read_all(self, embeddings: "np.ndarray") -> "np.ndarray":
        """Retrieve memories for a batch of embeddings."""

        if embeddings.shape != (self.num_nodes, self.embed_dim):
            raise ValueError("embedding shape mismatch")

        out = np.zeros_like(embeddings)
        for v in range(self.num_nodes):
            out[v] = self.read(v, embeddings[v])
        return out

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
