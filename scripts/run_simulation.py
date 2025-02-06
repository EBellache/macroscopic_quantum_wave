# quantum_wave_packet/scripts/run_simulation.py
import argparse
import jax.numpy as jnp
from quantum_wave_packet.core.grid import Grid1D
from quantum_wave_packet.core.solver import Solver1D
from quantum_wave_packet.core.potential import external_potential
from quantum_wave_packet.visualization.animate_wave_packet import WavePacketVisualizer1D


def run_simulation(num_steps=100, dt=0.01, omega=1.0, visualize=True):
    """
    Runs the quantum-like wave packet simulation.

    Parameters:
    - num_steps: Number of time steps to run the simulation.
    - dt: Time step size.
    - omega: Frequency of harmonic potential.
    - visualize: Whether to display real-time visualization.
    """

    # Setup grid and solver
    grid = Grid1D(x_min=-5, x_max=5, num_points=100)
    hbar_hat = 1.0  # Effective quantum parameter
    mass = 1.0  # Effective mass in wave dynamics

    solver = Solver1D(dt=dt, grid=grid, hbar_hat=hbar_hat, mass=mass)

    # Define external potential (harmonic oscillator)
    V_ext = lambda x: external_potential(x, type="harmonic", params={"omega": omega})

    # Initialize state: Gaussian wave packet
    rho = jnp.exp(-((grid.x) ** 2) / (2 * 1.0 ** 2))  # Initial Gaussian
    v = jnp.zeros_like(grid.x)
    state = {"x": grid.x, "rho": rho, "v": v}

    # Run simulation with or without visualization
    if visualize:
        visualizer = WavePacketVisualizer1D(grid)

        def update_state():
            """Update the simulation state at each step."""
            nonlocal state
            state = solver.update(state, V_ext)
            visualizer.state = state

        visualizer.start(update_state)

    else:
        print("Running simulation without visualization...")
        for _ in range(num_steps):
            state = solver.update(state, V_ext)

        # Output final density distribution
        print("Final density distribution:\n", state["rho"])


if __name__ == "__main__":
    # Command-line arguments for debugging/testing
    parser = argparse.ArgumentParser(description="Run the quantum-like wave packet simulation.")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of simulation time steps.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size.")
    parser.add_argument("--omega", type=float, default=1.0, help="Harmonic oscillator frequency.")
    parser.add_argument("--no_visual", action="store_true", help="Disable real-time visualization.")

    args = parser.parse_args()

    # Run the simulation
    run_simulation(num_steps=args.num_steps, dt=args.dt, omega=args.omega, visualize=not args.no_visual)
