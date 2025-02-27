import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)  # change this if you have gpu

# Parameters
size = 100  # grid size
steps = 5000  # number of growth steps
eta = 1.0  # eta -> determines the shape of the object
omega = 1.8  # relaxation constant

grid = ti.field(dtype=ti.i32, shape=(size, size))  # 2D Grid
concentration = ti.field(dtype=ti.f32, shape=(size, size))  # diffusion field
growth_candidates = ti.Vector.field(2, dtype=ti.i32, shape=size * size)
candidate_count = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def initialize_grid():
    """
    Initialize the grid with a seed at the center.
    """
    for i, j in grid:
        grid[i, j] = 0  # Empty space
        concentration[i, j] = 0.1  # SMALL initial diffusion everywhere

    grid[size // 2, size // 2] = 1  # locating a seed at the center
    concentration[size // 2, size // 2] = 1.0  # high initial concentration at seed


@ti.data_oriented
class SuccessiveOverRelaxation:
    """
    Solve the diffusion equation by proportionally taking the neighbouring
    values from the current state and the current value in place. Uses a
    checkerboard pattern to prevent race conditions. For "over relaxation",
    the value `self.omega` can be made greater than `1,0`.

    # Inputs
    - `threshold`: The minimal amount change needed for `self.run()` to keep
    iterating. Default `threshold = 1e-5`
    - `N`: The size of the grid. Default `N = 50`
    - `omega`: The relaxation constant. Default `omega = 1.8`
    """

    def __init__(self, omega=1.8, threshold=1e-5, max_iterations=200):
        self.omega = omega
        self.threshold = threshold
        self.max_iterations = max_iterations

    @ti.kernel
    def sor_iteration(self):
        for i, j in ti.ndrange((1, size - 1), (1, size - 1)):
            if grid[i, j] == 0:  # only update non cluster points
                new_value = (
                    concentration[i - 1, j]
                    + concentration[i + 1, j]
                    + concentration[i, j - 1]
                    + concentration[i, j + 1]
                ) * 0.25
                concentration[i, j] = (1 - self.omega) * concentration[
                    i, j
                ] + self.omega * new_value

    def solve(self, iterations=10):
        for _ in range(iterations):
            self.sor_iteration()


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
    Calculate the growth probabilities based on diffusion concentration.
    """
    total_prob = 0.0
    for k in range(candidate_count[None]):
        i, j = growth_candidates[k]
        probabilities[k] = concentration[i, j] ** eta  # use of diffusion concentration
        total_prob += probabilities[k]

    # Normalize probabilities
    if total_prob > 0:
        for k in range(candidate_count[None]):
            probabilities[k] /= total_prob
    else:
        for k in range(candidate_count[None]):
            probabilities[k] = 1.0 / candidate_count[None]  # uniform fallback


def simulate_dla():
    """
    Runs the DLA growth with SOR optimization.
    """
    sor_solver = SuccessiveOverRelaxation(omega=omega)
    sor_solver.solve(50)  # stabilize concentration field

    for step in range(steps):
        get_growth_candidates()
        num_candidates = candidate_count[None]

        if num_candidates == 0:
            break  # stop if there are no candidates left

        probabilities = np.zeros(num_candidates, dtype=np.float32)
        compute_growth_probabilities(probabilities)

        # choose random candidate
        chosen_index = np.random.choice(num_candidates, p=probabilities)
        i, j = growth_candidates.to_numpy()[chosen_index]
        grid[i, j] = 1  # grow the cluster

        # update every 10 steps
        if step % 10 == 0:
            sor_solver.solve(10)


def plot_grid():
    """
    Visualize the DLA cluster.
    """
    np_grid = grid.to_numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid, cmap="gray", origin="lower")
    plt.title("DLA growth simulation with SOR")
    plt.show()


# Run the simulation
initialize_grid()
simulate_dla()
plot_grid()
