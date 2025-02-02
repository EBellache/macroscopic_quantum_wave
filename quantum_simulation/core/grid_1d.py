# quantum_simulation/core/grid_1d.py
import jax.numpy as jnp


class Grid1D:
    def __init__(self, x_min, x_max, num_points):
        """Initialize a 1D grid."""
        self.x = jnp.linspace(x_min, x_max, num_points)
        self.dx = (x_max - x_min) / (num_points - 1)

    def gradient(self, f):
        """Compute first derivative using central differences."""
        return (jnp.roll(f, -1) - jnp.roll(f, 1)) / (2 * self.dx)

    def laplacian(self, f):
        """Compute second derivative using central differences."""
        return (jnp.roll(f, -1) - 2 * f + jnp.roll(f, 1)) / (self.dx ** 2)
