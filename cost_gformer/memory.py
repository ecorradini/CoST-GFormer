"""Memory buffers for CoST-GFormer.

The memory system is separated into three conceptual blocks:

- :class:`UltraShortTermAttention` (USTA): a very small buffer for the most
  recent tokens used directly by attention layers.
- :class:`ShortTermMemory` (STM): holds recent context for quick retrieval.
- :class:`LongTermMemory` (LTM): archives older information for long-range
  dependencies.

These classes are placeholders and only document the intended behavior.
"""

import numpy as np  # for type hints
import torch


class UltraShortTermAttention:
    """Represents the USTA buffer."""

    def __init__(self, size: int = 4):
        self.size = size

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"{self.__class__.__name__}(size={self.size})"


class ShortTermMemory:
    """Cyclic buffer storing recent node embeddings."""

    def __init__(self, size: int = 128, num_nodes: int | None = None, embed_dim: int | None = None, device: str | torch.device = "cpu") -> None:
        self.size = int(size)
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.embed_dim = None if embed_dim is None else int(embed_dim)
        self.device = torch.device(device)
        self._ptr = 0
        self._filled = 0
        self.buffer: torch.Tensor | None = None

        if self.num_nodes is not None and self.embed_dim is not None:
            self._init_buffer()

    # ------------------------------------------------------------------
    def _init_buffer(self) -> None:
        self.buffer = torch.zeros(
            (self.size, self.num_nodes, self.embed_dim), dtype=torch.float32, device=self.device
        )

    # ------------------------------------------------------------------
    def write(self, embeddings: "np.ndarray | torch.Tensor") -> None:
        """Store a new set of node embeddings."""

        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        if self.buffer is None:
            self.num_nodes, self.embed_dim = embeddings.shape
            self._init_buffer()
        if embeddings.shape != (self.num_nodes, self.embed_dim):
            raise ValueError("embedding shape mismatch")

        self.buffer[self._ptr] = embeddings.to(self.device)
        self._ptr = (self._ptr + 1) % self.size
        self._filled = min(self._filled + 1, self.size)

    # ------------------------------------------------------------------
    def read(self, node: int) -> "np.ndarray":
        """Return the mean embedding for ``node`` across the buffer."""

        if self.buffer is None or self._filled == 0:
            return torch.zeros(self.embed_dim, dtype=torch.float32).cpu().numpy()

        data = self.buffer[: self._filled, node]
        return data.mean(dim=0).cpu().numpy()

    def read_all(self) -> "np.ndarray":
        """Return mean embeddings for all nodes."""

        if self.buffer is None or self._filled == 0:
            return torch.zeros((self.num_nodes, self.embed_dim), dtype=torch.float32).cpu().numpy()

        data = self.buffer[: self._filled]
        return data.mean(dim=0).cpu().numpy()

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
        device: str | torch.device = "cpu",
    ) -> None:

        self.num_nodes = int(num_nodes)
        self.embed_dim = int(embed_dim)
        self.num_centroids = int(num_centroids)
        self.buffer_size = int(buffer_size)

        rng = np.random.default_rng() if rng is None else rng
        self.rng = rng
        self.device = torch.device(device)

        self.buffer = torch.zeros(
            (self.buffer_size, self.num_nodes, self.embed_dim), dtype=torch.float32, device=self.device
        )
        self._ptr = 0
        self._filled = 0

        self.centroids = torch.zeros(
            (self.num_nodes, self.num_centroids, self.embed_dim), dtype=torch.float32, device=self.device
        )
        # Gating projection for fuse()
        self.W_g = (
            torch.from_numpy(rng.standard_normal((2 * self.embed_dim, self.embed_dim), dtype=np.float32))
            / torch.sqrt(torch.tensor(2 * self.embed_dim, dtype=torch.float32))
        ).to(self.device)

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(x: torch.Tensor) -> torch.Tensor:
        x = x - x.max()
        e = torch.exp(x)
        return e / e.sum()

    @staticmethod
    def _sigmoid(x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-x))

    # ------------------------------------------------------------------
    def write(self, embeddings: "np.ndarray | torch.Tensor") -> None:
        """Store new embeddings in the circular buffer."""

        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        if embeddings.shape != (self.num_nodes, self.embed_dim):
            raise ValueError("embedding shape mismatch")

        self.buffer[self._ptr] = embeddings.to(self.device)
        self._ptr = (self._ptr + 1) % self.buffer_size
        self._filled = min(self._filled + 1, self.buffer_size)

    # ------------------------------------------------------------------
    def build(self, embeddings: "np.ndarray | torch.Tensor | None" = None, iters: int = 10) -> None:
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
        elif isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)

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
            centers = X[idx].clone()

            for _ in range(max(1, iters)):
                # assign points to nearest centroid
                dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2)
                assign = dists.argmin(dim=1)
                # update centroids
                for k in range(self.num_centroids):
                    mask = assign == k
                    if mask.any():
                        centers[k] = X[mask].mean(dim=0)

            self.centroids[v] = centers

    # ------------------------------------------------------------------
    def read(self, node: int, embedding: "np.ndarray | torch.Tensor") -> "np.ndarray":
        """Retrieve context vector for a node embedding."""

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).to(self.device)
        cents = self.centroids[node]
        sims = embedding @ cents.T
        weights = self._softmax(sims)
        return (weights @ cents).cpu().numpy()

    def read_all(self, embeddings: "np.ndarray | torch.Tensor") -> "np.ndarray":
        """Retrieve memories for a batch of embeddings."""

        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        if embeddings.shape != (self.num_nodes, self.embed_dim):
            raise ValueError("embedding shape mismatch")

        out = torch.zeros_like(embeddings)
        for v in range(self.num_nodes):
            out[v] = torch.from_numpy(self.read(v, embeddings[v]))
        return out.cpu().numpy()

    def fuse(self, node: int, embedding: "np.ndarray | torch.Tensor") -> "np.ndarray":
        """Fuse current embedding with retrieved memory."""

        if isinstance(embedding, np.ndarray):
            embedding_t = torch.from_numpy(embedding).to(self.device)
        else:
            embedding_t = embedding.to(self.device)
        mem = torch.from_numpy(self.read(node, embedding_t)).to(self.device)
        gate = self._sigmoid(torch.cat([embedding_t, mem]) @ self.W_g)
        fused = gate * embedding_t + (1.0 - gate) * mem
        return fused.cpu().numpy()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return (
            f"{self.__class__.__name__}(nodes={self.num_nodes}, "
            f"dim={self.embed_dim}, centroids={self.num_centroids})"
        )
