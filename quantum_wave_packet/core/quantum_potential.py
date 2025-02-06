# quantum_wave_packet/core/quantum_potential.py
import jax.numpy as jnp


def compute_quantum_potential(rho, grid, hbar_hat, mass):
    """Computes quantum-like potential Q(x,t) from probability density rho."""
    sqrt_rho = jnp.sqrt(rho)
    laplacian_sqrt_rho = grid.laplacian(sqrt_rho)
    Q = - (hbar_hat ** 2 / (2 * mass)) * laplacian_sqrt_rho / sqrt_rho
    return Q
