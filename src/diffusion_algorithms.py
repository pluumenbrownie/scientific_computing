import numpy as np
import taichi as ti
import math as mt


ti.init(arch=ti.cpu)  # change this if you have gpu


# Parameters
D = 1.0  # diffusion coefficient
N = 50  # gridpoints
OMEGA = 1.8  # relaxation constant
dx = 1.0 / N  # gridspacing
dt = 0.25 * dx**2 / D  # stability condition
total_time = 1.0  # total simulation time
num_steps = int(total_time / dt)  # number of time steps
times = [0, 0.001, 0.01, 0.1, 1.0]  # different times

image_scale = 4

# Initialize grid
concentration = ti.field(float, shape=(N, N))
c_difference = ti.field(float, shape=(N, N))
white_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.ceil(N**2 / 2)))
black_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.floor(N**2 / 2)))


@ti.func
def pbi(i: int) -> int:
    """
    Convert normal coordinates into periodic boundary coordinates.
    """
    if i < 0:
        i += N
    if i >= N:
        i -= N
    return i


@ti.kernel
def init_concentration():
    for i, j in concentration:
        if j == 0:  # boundary condition
            concentration[i, j] = 0
        elif j == N - 1:  # boundary condition
            concentration[i, j] = 1
        else:
            concentration[i, j] = 0


@ti.func
def neighbourhood_values(i: int, j: int, field) -> float:
    """
    Get the combined values of the four neighbours of the given cell,
    using boundary conditions.
    """
    return (
        field[pbi(i - 1), j]
        + field[pbi(i + 1), j]
        + field[pbi(i), j - 1]
        + field[pbi(i), j + 1]
    )


@ti.func
def copy_into_difference():
    for i, j in concentration:
        c_difference[i, j] = concentration[i, j]


@ti.kernel
def solve_jacobi():
    copy_into_difference()
    for i, j in concentration:
        if j == 0 or j >= N - 1:
            continue
        concentration[i, j] = 0.25 * neighbourhood_values(i, j, c_difference)
    reset_boundary()
    calculate_differences()


@ti.func
def calculate_differences():
    for i, j in c_difference:
        c_difference[i, j] = abs(concentration[i, j] - c_difference[i, j])


@ti.func
def reset_boundary():
    for i in range(N):
        concentration[i, 0] = 1.0
        concentration[i, N - 1] = 0.0


def run_gui(scale: int = 1):
    gui = ti.GUI("Vibrating String", res=(scale * N, scale * N))  # type:ignore
    image = concentration.to_numpy()
    if not scale == 1:
        scaled_image = np.zeros(shape=(scale * N, scale * N))
        for i, j in np.ndindex(scaled_image.shape):
            scaled_image[i, j] = image[i // scale, j // scale]
        image = scaled_image
    image = np.flip(image, axis=1)
    while gui.running:
        gui.set_image(image)
        gui.show()


def run_jacobi(threshold: float = 1e-5):
    init_concentration()
    solve_jacobi()
    while c_difference.to_numpy().max() > threshold:
        solve_jacobi()
    run_gui(scale=8)


@ti.kernel
def init_checkerboard():
    amount = 0
    for i, j in concentration:
        # for i, j in np.ndindex(concentration.shape):
        if i % 2 == j % 2:
            white_tiles[j // 2 + ti.ceil(i * N / 2, dtype=int)] = ti.Vector([i, j])
            amount += 1
        else:
            black_tiles[j // 2 + ti.floor(i * N / 2, dtype=int)] = ti.Vector([i, j])


@ti.kernel
def solve_gauss_seidel():
    copy_into_difference()
    for v in black_tiles:
        i, j = int(black_tiles[v][0]), int(black_tiles[v][1])
        if j == 0 or j >= N - 1:
            continue
        concentration[i, j] = 0.25 * neighbourhood_values(i, j, concentration)

    for v in white_tiles:
        i, j = int(white_tiles[v][0]), int(white_tiles[v][1])
        if j == 0 or j >= N - 1:
            continue
        concentration[i, j] = 0.25 * neighbourhood_values(i, j, concentration)
    calculate_differences()


def run_gauss_seidel(threshold: float = 1e-5):
    init_checkerboard()
    init_concentration()
    solve_gauss_seidel()
    while c_difference.to_numpy().max() > threshold:
        solve_gauss_seidel()
    run_gui(scale=10)


@ti.kernel
def solve_succesive_over_relaxation():
    copy_into_difference()
    for v in black_tiles:
        i, j = int(black_tiles[v][0]), int(black_tiles[v][1])
        if j == 0 or j >= N - 1:
            continue
        concentration[i, j] = (
            OMEGA * 0.25 * neighbourhood_values(i, j, concentration)
            + (1 - OMEGA) * concentration[i, j]
        )

    for v in white_tiles:
        i, j = int(white_tiles[v][0]), int(white_tiles[v][1])
        if j == 0 or j >= N - 1:
            continue
        concentration[i, j] = (
            OMEGA * 0.25 * neighbourhood_values(i, j, concentration)
            + (1 - OMEGA) * concentration[i, j]
        )
    calculate_differences()


def run_SOR(threshold: float = 1e-5):
    init_checkerboard()
    init_concentration()
    solve_succesive_over_relaxation()
    while c_difference.to_numpy().max() > threshold:
        solve_succesive_over_relaxation()
    run_gui(scale=10)


if __name__ == "__main__":
    # run_jacobi()
    # run_gauss_seidel()
    run_SOR()
