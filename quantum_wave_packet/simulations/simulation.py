# quantum_wave_packet/scripts/run_simulation.py
from quantum_wave_packet.core.grid import Grid1D
from quantum_wave_packet.core.solver import Solver1D
from quantum_wave_packet.core.potential import external_potential
from quantum_wave_packet.visualization.animate_wave_packet import WavePacketVisualizer1D
import jax.numpy as jnp

# Initialize grid and solver parameters
grid = Grid1D(x_min=-5, x_max=5, num_points=100)
dt = 0.01
hbar_hat = 1.0  # Neuronal quantum constant
mass = 1.0  # Effective mass from LC circuit
omega = 1.0  # Frequency of subthreshold oscillation

solver = Solver1D(dt=dt, grid=grid, hbar_hat=hbar_hat, mass=mass)

# Define external potential (harmonic for LC circuit analogy)
V_ext = lambda x: external_potential(x, type="harmonic", params={"omega": omega})

# Initial state: Gaussian centered at resting potential
rho = jnp.exp(-((grid.x) ** 2) / (2 * 1.0 ** 2))  # Initial Gaussian profile
v = jnp.zeros_like(grid.x)
state = {"x": grid.x, "rho": rho, "v": v}

# Create visualizer
visualizer = WavePacketVisualizer1D(grid)


def update_state():
    """Updates the simulation state in real-time."""
    global state
    state = solver.update(state, V_ext)
    visualizer.state = state  # Send updated state to visualization


# Start visualization
visualizer.start(update_state)
