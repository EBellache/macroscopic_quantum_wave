import tkinter as tk
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class WavePacketVisualizer1D:
    """Tkinter-based real-time visualization for the wave packet simulation."""

    def __init__(self, grid, update_interval=50):
        """
        Initializes the visualization.

        Parameters:
        - grid: Grid1D object containing spatial points
        - update_interval: Time in milliseconds between updates
        """
        self.grid = grid
        self.update_interval = update_interval

        # Setup Tkinter window
        self.root = tk.Tk()
        self.root.title("1D Wave Packet Evolution")

        # Create Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.line, = self.ax.plot(grid.x, np.zeros_like(grid.x), 'b-', lw=2)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(grid.x.min(), grid.x.max())
        self.ax.set_xlabel("Position x")
        self.ax.set_ylabel("Density ρ")

        # Embed Matplotlib in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Simulation state
        self.state = None
        self.running = False

    def update_plot(self):
        """Updates the 1D plot based on the current state."""
        if self.state:
            self.line.set_ydata(np.array(self.state["rho"]))  # Convert JAX array to NumPy
            self.canvas.draw()

        if self.running:
            self.root.after(self.update_interval, self.update_plot)

    def start(self, state_updater):
        """
        Starts the 1D visualization.

        Parameters:
        - state_updater: Function that updates simulation state
        """
        self.state_updater = state_updater
        self.running = True
        self.update_plot()
        self.root.mainloop()

    def stop(self):
        """Stops the visualization."""
        self.running = False
        self.root.quit()
