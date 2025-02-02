import jax.numpy as jnp
from quantum_simulation.potentials.base_potential import BasePotential


class ExternalPotential(BasePotential):
    def __init__(self, V_func):
        """
        V_func: A callable function V(x) that defines the external potential
        """
        self.V_func = V_func

    def compute(self, rho, grid):
        """Apply external potential."""
        return self.V_func(grid.x)
