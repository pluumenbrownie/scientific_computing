import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)  # change this if you have gpu

# Parameters
size = 100  # Grid size
steps = 5000  # Number of growth steps
eta = 1.0  # Growth parameter

grid = ti.field(dtype=ti.i32, shape=(size, size))  # 2D Grid
growth_candidates = ti.Vector.field(2, dtype=ti.i32, shape=size * size)
candidate_count = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def initialize_grid():
    """
    Initialize the grid with a seed at the center.
    """
    for i, j in grid:
        grid[i, j] = 0  # empty space
    grid[size // 2, size // 2] = 1  # locating a seed at the center


@ti.kernel
def get_growth_candidates():
    """
    Identify the locations of the candidates adjacent to the cluster.
    """
    candidate_count[None] = 0
    for i, j in ti.ndrange((1, size - 1), (1, size - 1)):
        if grid[i, j] == 0 and (
            grid[i - 1, j] == 1
            or grid[i + 1, j] == 1
            or grid[i, j - 1] == 1
            or grid[i, j + 1] == 1
        ):
            idx = ti.atomic_add(candidate_count[None], 1)
            growth_candidates[idx] = ti.Vector([i, j])


@ti.kernel
def compute_growth_probabilities(probabilities: ti.types.ndarray()):
    """
    Calculate the growth probabilities.
    """
    total_prob = 0.0
    for k in range(candidate_count[None]):
        probabilities[k] = ti.random(ti.f32) ** eta
        total_prob += probabilities[k]

    # Normalize probabilities
    for k in range(candidate_count[None]):
        probabilities[k] /= total_prob


def simulate_dla():
    """
    Runs the DLA growth.
    """
    for _ in range(steps):
        get_growth_candidates()
        num_candidates = candidate_count[None]

        if num_candidates == 0:
            break  # stop if there are no candidates left

        probabilities = np.zeros(num_candidates, dtype=np.float32)
        compute_growth_probabilities(probabilities)

        # choose a candidate random based on probabilities
        chosen_index = np.random.choice(num_candidates, p=probabilities)
        i, j = growth_candidates.to_numpy()[chosen_index]
        grid[i, j] = 1  # grow the cluster


def plot_grid():
    """
    Visualize the DLA cluster.
    """
    np_grid = grid.to_numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid, cmap="gray", origin="lower")
    plt.title("DLA Growth Simulation")
    plt.show()


# Run the simulation
initialize_grid()
simulate_dla()
plot_grid()
