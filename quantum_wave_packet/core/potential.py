# quantum_wave_packet/core/potential.py
import jax.numpy as jnp


def external_potential(x, type="harmonic", params={}):
    """Computes external potential V(x)."""
    if type == "harmonic":
        omega = params.get("omega", 1.0)
        return 0.5 * omega ** 2 * x ** 2
    elif type == "free":
        return jnp.zeros_like(x)  # No external potential
    else:
        raise ValueError("Unknown potential type")
