# quantum_simulation/core/grid_2d.py
import jax.numpy as jnp


class Grid2D:
    def __init__(self, x_min, x_max, y_min, y_max, num_x, num_y):
        """Initialize a 2D grid."""
        self.x = jnp.linspace(x_min, x_max, num_x)
        self.y = jnp.linspace(y_min, y_max, num_y)
        self.dx = (x_max - x_min) / (num_x - 1)
        self.dy = (y_max - y_min) / (num_y - 1)

    def gradient(self, f):
        """Compute first derivative using central differences in both directions."""
        dfdx = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * self.dx)
        dfdy = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * self.dy)
        return dfdx, dfdy

    def laplacian(self, f):
        """Compute second derivative using central differences."""
        d2fdx2 = (jnp.roll(f, -1, axis=1) - 2 * f + jnp.roll(f, 1, axis=1)) / (self.dx ** 2)
        d2fdy2 = (jnp.roll(f, -1, axis=0) - 2 * f + jnp.roll(f, 1, axis=0)) / (self.dy ** 2)
        return d2fdx2 + d2fdy2
