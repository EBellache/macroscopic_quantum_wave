from abc import ABC, abstractmethod


class BasePotential(ABC):
    @abstractmethod
    def compute(self, rho, grid):
        pass
