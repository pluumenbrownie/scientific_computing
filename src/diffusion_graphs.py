import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc
import taichi as ti
from diffusion_algorithms import Jacobi, GaussSeidel, SuccessiveOverRelaxation


DIFFUSION_CONSTANT = 1.0
N = 50   # gridpoints
dx = 1.0 / N  # gridspacing
dt = 0.15 * dx**2 / DIFFUSION_CONSTANT  # stability condition
total_time = 1.0  # total simulation time
num_steps = int(total_time / dt)  # number of time steps
times = [0, 0.001, 0.01, 0.1, 1.0]  # different times


def analytical_solution(x: NDArray, t: float):
    if t == 0:
        return np.zeros_like(x)
    sum_terms = np.zeros_like(x)
    for i in range(50):
        sum_terms += erfc(
            (1 - x + 2 * i) / (2 * np.sqrt(DIFFUSION_CONSTANT * t))
        ) - erfc((1 + x + 2 * i) / (2 * np.sqrt(DIFFUSION_CONSTANT * t)))
    return sum_terms


def compare_to_analytical():
    y_values = np.linspace(0, 1, N)
    fig, ax = plt.subplots()
    analytical = analytical_solution(y_values, 1.0)
    ax.plot(y_values, analytical, "--", label="Analytical t=1.0")
    
    jacobi = Jacobi(N=N)
    jacobi.run()
    jacobi_result = jacobi.concentration.to_numpy()[N // 2, :]
    ax.plot(y_values, jacobi_result, ".", label=f"Jacobi Iteration")

    gauss = GaussSeidel(N=N)
    gauss.run()
    gauss_result = gauss.concentration.to_numpy()[N // 2, :]
    ax.plot(y_values, gauss_result, ".", label=f"Gauss-Seidel Iteration")

    sov = SuccessiveOverRelaxation(omega=1.8, N=N)
    sov.run()
    sov_result = sov.concentration.to_numpy()[N // 2, :]
    ax.plot(y_values, sov_result, ".", label=f"Succesive Over Relaxation")

    ax.set_xlabel("y")
    ax.set_ylabel("c(y)")
    ax.legend()
    ax.set_title("Comparison of numerical and analytical solutions")
    plt.savefig("local/numerical_analytical.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    compare_to_analytical()
