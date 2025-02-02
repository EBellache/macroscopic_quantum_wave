import jax.numpy as jnp
from quantum_simulation.potentials.base_potential import BasePotential


class QuantumPotential(BasePotential):
    def __init__(self, D):
        self.D = D

    def compute(self, rho, grid):
        """Compute the quantum potential"""
        sqrt_rho = jnp.sqrt(rho)
        laplacian_rho = grid.laplacian(sqrt_rho)
        return -2 * self.D ** 2 * laplacian_rho / sqrt_rho
