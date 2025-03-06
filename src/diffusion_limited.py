import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import math as mt

# from diffusion_algorithms import SuccessiveOverRelaxation

ti.init(arch=ti.cpu)  # change this if you have gpu

# Parameters
size = 100  # grid size
steps = 500  # number of growth steps
eta = 1.2  # eta -> determines the shape of the object
omega = 1.92  # reaxation constant

concentration = ti.field(dtype=ti.f32, shape=(size, size))  # diffusion field
change_in_concentration = ti.field(dtype=ti.f32, shape=(size, size))  # diffusion field
growth_candidates = ti.Vector.field(2, dtype=ti.i32, shape=size * size)
candidate_count = ti.field(dtype=ti.i32, shape=())
probabilities = ti.field(dtype=ti.f32, shape=(size * size))
chosen_index = ti.field(dtype=ti.i32, shape=())
grid = ti.field(dtype=float, shape=(size, size))  # 2D grid
white_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.ceil(size**2 / 2)))
black_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.floor(size**2 / 2)))


@ti.kernel
def init_checkerboard():
    """
    Fills the checkerboard fields `self.white_tiles` and `self.black_tiles`.
    """
    for i, j in concentration:
        if i % 2 == j % 2:
            white_tiles[j // 2 + ti.ceil(i * size / 2, dtype=int)] = ti.Vector([i, j])
        else:
            black_tiles[j // 2 + ti.floor(i * size / 2, dtype=int)] = ti.Vector([i, j])


init_checkerboard()


@ti.kernel
def initialize_grid():
    """
    Initialize the grid with a seed at the center.
    """
    for i, j in ti.ndrange(size, size):
        grid[i, j] = 0  # Empty space
        if i == size - 1:
            concentration[i, j] = 1

    grid[0, size // 2] = 1  # placing the seed at the bottom of the grid


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

    def __init__(
        self,
        concentration,
        change_in_concentration,
        omega=1.8,
        threshold=1e-5,
        max_iterations=200,
    ):
        self.omega = omega
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.concentration = concentration
        self.change_in_concentration = change_in_concentration

    @ti.kernel
    def sor_iteration(self) -> float:
        # return largest change to determine whether theshold is reached
        largest_change = 0.0
        for tile in white_tiles:
            i, j = white_tiles[tile][0], white_tiles[tile][1]
            # do not update the sources and drains at the boundaries
            if i == size - 1 or i == 0:
                continue
            if grid[i, j] == 0:  # only update non cluster points
                change = self.update_cell(i, j)
                largest_change = max(change, largest_change)
        for tile in black_tiles:
            i, j = black_tiles[tile][0], black_tiles[tile][1]
            # do not update the sources and drains at the boundaries
            if i == size - 1 or i == 0:
                continue
            if grid[i, j] == 0:  # only update non cluster points
                change = self.update_cell(i, j)
                largest_change = max(change, largest_change)

        return largest_change

    @ti.func
    def copy_into_change(self):
        for i, j in self.change_in_concentration:
            self.change_in_concentration[i, j] = self.concentration[i, j]

    @ti.func
    def calculate_change(self):
        for i, j in self.change_in_concentration:
            self.change_in_concentration[i, j] = abs(
                self.concentration[i, j] - self.change_in_concentration[i, j]
            )

    @ti.func
    def update_cell(self, i, j) -> float:
        old_value = self.concentration[i, j]
        new_value = (
            self.concentration[i - 1, j]
            + self.concentration[i + 1, j]
            + self.concentration[i, periodic_boundary(j - 1)]  # periodic boundary
            + self.concentration[i, periodic_boundary(j + 1)]  # periodic boundary
        ) * 0.25
        self.concentration[i, j] = (
            self.omega * new_value + (1 - self.omega) * self.concentration[i, j]
        )
        return abs(old_value - self.concentration[i, j])

    def solve(self, iterations=10):
        largest_change = self.sor_iteration()
        while largest_change > self.threshold:
            largest_change = self.sor_iteration()


@ti.func
def periodic_boundary(i: int):
    """
    Assign periodic boundary to the grid
    """
    return i % size


@ti.kernel
def get_growth_candidates():
    """
    Identify the locations of the candidates adjacent to the cluster.
    """
    candidate_count[None] = 0
    for i, j in ti.ndrange((0, size), (0, size)):
        if grid[i, j] == 0 and (
            grid[i - 1, j] == 1
            or grid[i + 1, j] == 1
            or grid[i, periodic_boundary(j - 1)] == 1  # periodic boundary
            or grid[i, periodic_boundary(j + 1)] == 1  # periodic boundary
        ):
            idx = ti.atomic_add(candidate_count[None], 1)
            growth_candidates[idx] = ti.Vector([i, j])


@ti.kernel
def compute_growth_probabilities():
    """
    Calculate the growth probabilities based on diffusion concentration.
    """
    for i in probabilities:
        probabilities[i] = 0.0
    total_prob = 0.0
    for k in range(candidate_count[None]):
        i, j = growth_candidates[k]
        probabilities[k] = concentration[i, j] ** eta
        if ti.math.isnan(probabilities[k]):  # Check for NaN
            probabilities[k] = 1e-10
        total_prob += probabilities[k]

    # Normalize probabilities
    if total_prob > 0:
        for k in range(candidate_count[None]):
            probabilities[k] /= total_prob
    else:
        for k in range(candidate_count[None]):
            probabilities[k] = 1.0 / candidate_count[None]  # uniform fallback


@ti.kernel
def choose_site():
    """
    Select a site to grow the DLA based on the growth probability
    """
    cum = 0.0
    flag = 0
    U = ti.random(ti.f32)

    ti.loop_config(serialize=True)
    for k in range(candidate_count[None]):
        if flag == 0:
            cum += probabilities[k]
            if U <= cum:
                chosen_index[None] = k
                flag = 1  # turn the flag to stop adding the probability
                break

    if flag == 0:
        chosen_index[None] = candidate_count[None] - 1  # fall back


def simulate_dla():
    """
    Runs the DLA growth with SOR optimization.
    """
    initialize_grid()
    sor_solver = SuccessiveOverRelaxation(
        concentration, change_in_concentration, omega=omega
    )
    sor_solver.solve(100)
    for step in range(steps):
        get_growth_candidates()
        num_candidates = candidate_count[None]

        if num_candidates == 0:
            break  # stop if there are no candidates left

        compute_growth_probabilities()
        choose_site()
        i, j = growth_candidates[chosen_index[None]]
        grid[i, j] = 1  # grow the cluster
        concentration[i, j] = 0.0

        # update every 10 steps
        sor_solver.solve(100)


def plot_grid():
    """
    Visualize the DLA cluster.
    """
    np_grid = grid.to_numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid, cmap="gray", origin="lower")
    plt.title("DLA growth simulation with SOR")
    plt.show()


def plot_concentration_and_dla():
    """
    Plots the concentration field and overlays the DLA cluster.
    """
    grid_np = grid.to_numpy()
    concentration_np = concentration.to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = plt.cm.plasma
    im = ax.imshow(concentration_np, cmap=cmap, origin="lower")

    dla_mask = np.ma.masked_where(grid_np == 0, grid_np)
    ax.imshow(dla_mask, cmap="gray", alpha=0.8, origin="lower")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Diffusion Concentration")

    ax.set_title(f"DLA Growth with SOR Concentration Field with $\\eta=${eta}" "" "")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.savefig("local/dla.png")
    plt.show()


# Run the simulation
simulate_dla()
plot_concentration_and_dla()
