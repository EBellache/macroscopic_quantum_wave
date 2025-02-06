# quantum_wave_packet/core/solver.py
import jax.numpy as jnp
from jax import jit
from quantum_simulation.core.quantum_potential import compute_quantum_potential


class Solver1D:
    """1D Solver for macroscopic quantum-like wave packet dynamics."""

    def __init__(self, dt, grid, hbar_hat, mass):
        self.dt = dt
        self.grid = grid
        self.hbar_hat = hbar_hat
        self.mass = mass

    @jit
    def update(self, state, external_potential):
        """Updates the state using the Lax-Friedrichs method."""
        x, rho, v = state["x"], state["rho"], state["v"]
        dx = self.grid.dx

        # Compute potentials
        V_ext = external_potential(x)
        Q = compute_quantum_potential(rho, self.grid, self.hbar_hat, self.mass)

        # Effective potential
        V_eff = V_ext + Q

        # Compute force
        F_eff = -self.grid.gradient(V_eff)

        # Update velocity
        v_new = v + self.dt * F_eff / self.mass

        # Update density using Lax-Friedrichs scheme
        rho_shifted = (jnp.roll(rho, 1) + jnp.roll(rho, -1)) / 2
        v_shifted = (jnp.roll(v_new, 1) + jnp.roll(v_new, -1)) / 2
        rho_new = rho_shifted - self.dt * self.grid.gradient(rho * v_shifted)

        # Apply Outflow BC (Neumann)
        rho_new = rho_new.at[0].set(rho_new[1])
        rho_new = rho_new.at[-1].set(rho_new[-2])
        v_new = v_new.at[0].set(v_new[1])
        v_new = v_new.at[-1].set(v_new[-2])

        return {"x": x, "rho": rho_new, "v": v_new}
