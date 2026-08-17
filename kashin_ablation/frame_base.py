from abc import ABC, abstractmethod


class BaseFrame(ABC):
    """统一的 Kashin frame 接口。"""

    def __init__(self, d, D):
        self.d = int(d)
        self.D = int(D)
        self.A = self.D / self.d

    @abstractmethod
    def analysis(self, x):
        """R^d -> R^D coefficient vector."""
        raise NotImplementedError

    @abstractmethod
    def synthesis(self, a):
        """R^D -> R^d reconstructed vector."""
        raise NotImplementedError

