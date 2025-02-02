# quantum_simulation/core/solver_1d.py
import jax.numpy as jnp


class Solver1D:
    def __init__(self, dt, grid):
        self.dt = dt
        self.grid = grid

    def update(self, state, potential):
        """Update using Lax-Friedrichs scheme."""
        x, rho, v = state["x"], state["rho"], state["v"]
        dx = self.grid.dx

        # Compute quantum force
        Q = potential.compute(rho, self.grid)
        FQ = -self.grid.gradient(Q)

        # Update velocity
        v_new = v + self.dt * FQ

        # Update density using Lax-Friedrichs
        rho_shifted = (jnp.roll(rho, 1) + jnp.roll(rho, -1)) / 2
        v_shifted = (jnp.roll(v_new, 1) + jnp.roll(v_new, -1)) / 2
        rho_new = rho_shifted - self.dt * self.grid.gradient(rho * v_shifted)

        return {"x": x, "rho": rho_new, "v": v_new}
