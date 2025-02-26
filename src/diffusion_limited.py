import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Tuple

def initialize_grid(size: np.int_ = 100) -> NDArray[np.int_]:
    """Create a square grid with a single seed at the center."""
    grid: NDArray[np.int_] = np.zeros((size, size), dtype=np.int_)
    grid[size // 2, size // 2] = 1  # Start with a seed at the center
    return grid

def get_growth_candidates(grid: NDArray[np.int_]) -> List[Tuple[np.int_, np.int_]]:
    """Identify candidates for growth adjacent to the cluster."""
    candidates: List[Tuple[np.int_, np.int_]] = []
    size: np.int_ = grid.shape[0]
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            if grid[i, j] == 0:
                if grid[i - 1, j] == 1 or grid[i + 1, j] == 1 or grid[i, j - 1] == 1 or grid[i, j + 1] == 1:
                    candidates.append((i, j))
    return candidates

def calculate_growth_probabilities(grid: NDArray[np.int_], candidates: List[Tuple[np.int_, np.int_]], eta: np.float64) -> NDArray[np.float64]:
    """Calculate the probability of growth for each candidate."""
    concentrations: NDArray[np.float64] = np.array([np.random.rand() for _ in candidates])
    probabilities: NDArray[np.float64] = concentrations ** eta
    probabilities /= np.sum(probabilities)  # Normalize to sum to 1
    return probabilities

def simulate_dla(grid: NDArray[np.int_], steps: np.int_ = 5000, eta: np.float64 = 1.0) -> NDArray[np.int_]:
    """Simulate the process."""
    for _ in range(steps):
        candidates: List[Tuple[np.int_, np.int_]] = get_growth_candidates(grid)
        if not candidates:
            break  # Stop if no growth candidates are left
        probabilities: NDArray[np.float64] = calculate_growth_probabilities(grid, candidates, eta)
        chosen_index: np.int_ = np.random.choice(len(candidates), p=probabilities)
        i, j = candidates[chosen_index]
        grid[i, j] = 1
    return grid

def plot_grid(grid: NDArray[np.int_]) -> None:
    """Visualize the DLA cluster."""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='gray', origin='lower')
    plt.title("DLA Growth Simulation")
    plt.show()

# Run the simulation
grid: NDArray[np.int_] = initialize_grid(size=100)
grid = simulate_dla(grid, steps=5000, eta=1.0)
plot_grid(grid)
