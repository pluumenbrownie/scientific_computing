import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc
from scipy.optimize import minimize_scalar
import taichi as ti
from diffusion_algorithms import Jacobi, GaussSeidel, SuccessiveOverRelaxation
from alive_progress import alive_bar

ti.init()


DIFFUSION_CONSTANT = 1.0
N = 50  # gridpoints
dx = 1.0 / N  # gridspacing
dt = 0.15 * dx**2 / DIFFUSION_CONSTANT  # stability condition
total_time = 1.0  # total simulation time
num_steps = int(total_time / dt)  # number of time steps
times = [0, 0.001, 0.01, 0.1, 1.0]  # different times
threshold = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]#, 1e-7, 1e-8]
omega = np.linspace(0.5, 1.95, 10)  # typical value for omega


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
    """
    Compare Jacobi, Gauss-Seidel and SOR result to the
    analytical ones
    """
    y_values = np.linspace(0, 1, N)
    analytical = analytical_solution(y_values, 1) 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    jacobi = Jacobi(N=N)
    jacobi.run()
    jacobi_result = jacobi.concentration.to_numpy()[N // 2,:]
    jacobi_result = jacobi_result[:, 0]
    axes[0].plot(y_values, analytical, "--", label="Analytical t=1.0", color="black")
    axes[0].plot(y_values, jacobi_result, "o", label="Jacobi", color="red")
    axes[0].set_title("Jacobi vs. Analytical")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("c(y)")
    axes[0].legend()
    axes[0].grid()

   
    gauss = GaussSeidel(N=N)
    gauss.run()
    gauss_result = gauss.concentration.to_numpy()[N // 2, :]
    gauss_result = gauss_result[:, 0]
    axes[1].plot(y_values, analytical, "--", label="Analytical t=1.0", color="black")
    axes[1].plot(y_values, gauss_result, "s", label="Gauss-Seidel", color="blue")
    axes[1].set_title("Gauss-Seidel vs. Analytical")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("c(y)")
    axes[1].legend()
    axes[1].grid()

    
    sov = SuccessiveOverRelaxation(omega=1.5, N=N)  # Adjusted omega
    sov.run()
    sov_result = sov.concentration.to_numpy()[N // 2, :]
    sov_result = sov_result[:, 0]
    axes[2].plot(y_values, analytical, "--", label="Analytical t=1.0", color="black")
    axes[2].plot(y_values, sov_result, "d", label="SOR", color="green")
    axes[2].set_title("SOR vs. Analytical")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("c(y)")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.savefig("local/numerical_analytical.png", dpi=300)
    plt.show()


def over_relaxation_variable():
    """
    Show iteration count for differing values of omega and different
    grid sizes.
    """
    plot_x = np.linspace(1.6, 1.99, 50)
    sizes = [50, 60, 70, 80, 90]
    with alive_bar(len(sizes) * len(plot_x)) as bar:
        for grid_size in sizes:
            bar.title(f"Size: {grid_size}, {sizes.index(grid_size) + 1}/{len(sizes)}")
            plot_y = []
            for omega in plot_x:
                sov = SuccessiveOverRelaxation(omega=omega, N=grid_size)
                iteration,_ = sov.run()
                plot_y.append(iteration)
                bar()

            plt.plot(plot_x, plot_y, "o", label=f"N={grid_size}")
    plt.legend()
    plt.title("SOV iterations for differing $\\omega$")
    plt.xlabel("$\\omega$")
    plt.ylabel("Iterations")
    plt.savefig("local/relaxation_variable.png")
    plt.show()
    plt.close()


def convergence_iteration():
    jacobi_delta = []
    gauss_delta = []
    sov_delta = []

    for th in threshold:
        print("Start simulation for Jacobi")
        jacobi = Jacobi(N=N, threshold=th)
        _, jacobi_deltas = jacobi.run()  # Return delta per iteration
        jacobi_delta.append(jacobi_deltas)

    for th in threshold:
        print("Start simulation for Gauss-Seidel")
        gauss = GaussSeidel(N=N, threshold=th)
        _, gauss_deltas = gauss.run()
        gauss_delta.append(gauss_deltas)

    for th in threshold:
        print("Start simulation for SOV")
        sov = SuccessiveOverRelaxation(omega=1.8, N=N, threshold=th)
        _, sov_deltas = sov.run()
        sov_delta.append(sov_deltas)

    print("Finish simulation")
    return jacobi_delta, gauss_delta, sov_delta


def compare_omega(omega):
    sov_list = []
    for o in omega:
        sov = SuccessiveOverRelaxation(omega=o, N=N, threshold=1e-5)
        iteration, _ = sov.run()
        sov_list.append(iteration)
    return sov_list


def plot_convergence_ite(jacobi, gauss, sov):
    threshold_log = -np.log10(threshold)  # log scale
    plt.plot(threshold_log, jacobi, "o-", color="red", label="Jacobi")
    plt.plot(threshold_log, gauss, "o-", color="green", label="Gauss-Seidel")
    plt.plot(threshold_log, sov, "o-", color="blue", label="SOV")
    plt.legend()
    plt.title("Threshold value vs. Iteration")
    plt.xlabel("P (threshold value = 10^(-P))")
    plt.ylabel("Iterations")
    plt.grid()
    plt.savefig("local/threshold_iterations.png", dpi=300)
    plt.show()


def plot_omega(sov):
    plt.plot(omega, sov, "o-")
    plt.xlabel("omega value")
    plt.ylabel("iteration")
    plt.title("SOR iteration with different omega values")
    plt.grid()
    plt.savefig("local/omega_value.png", dpi=300)
    plt.show()


def plot_convergence_delta(jacobi_data, gauss_data, sov_data):
    """
    Plots the convergence measure delta against iterations k for Jacobi, Gauss-Seidel, and SOR.
    """
    plt.figure(figsize=(10, 6))

    # Plot Jacobi method
    for id, j_data in enumerate(jacobi_data):
        if id == 0:
            label = "Jacobi"
        else:
            label = None
        plt.plot(range(len(j_data)), j_data, label=label, linestyle="dashed", color="red")
    

    # Plot Gauss-Seidel method
    for id, g_data in enumerate(gauss_data):
        if id == 0:
            label = "Gauss-Seidel"
        else:
            label = None
        plt.plot(range(len(g_data)),g_data,label=label,linestyle="dotted",color="green",)

    # Plot SOR method
    for id, s_data in enumerate(sov_data):
        if id == 0:
            label = "SOR(ω=1.8)"
        else:
            label = None
        plt.plot(range(len(s_data)), s_data, label=label, color="blue")

    plt.yscale("log")  # Logarithmic scale
    plt.xlabel("Iterations (k)")
    plt.ylabel("Convergence measure δ")
    plt.title("Convergence of Iterative Methods")
    plt.legend()
    plt.grid()
    plt.savefig("local/convergence_delta.png", dpi=300)
    plt.show()

    # Wait till 1.6J is written then this code can work
    # Plot
    optimal_omega_no_obstacles = find_optimal_omega()
    print(f"Optimal ω without obstacles: {optimal_omega_no_obstacles}")

    # Add obstacles and find new optimal omega
    sor_with_obstacles = SuccessiveOverRelaxation(N=N, omega=1.8)
    sor_with_obstacles.add_rectangle(15, 25, 15, 25)
    optimal_omega_with_obstacles = find_optimal_omega(sink = (15, 25, 15, 25))
    print(f"Optimal ω with obstacles: {optimal_omega_with_obstacles}")


def find_optimal_omega(
    start_omege: float = 1.9, sink: tuple[int, int, int, int] | None = None
) -> float:
    def run_sov_with(omega: float, sink=sink) -> float:
        sov = SuccessiveOverRelaxation(omega=omega)
        if sink is not None:
            sov.add_rectangle(*sink)
        runs, _ = sov.run()
        return runs

    res = minimize_scalar(run_sov_with, bounds=[1.7, 1.98])
    return res.x


if __name__ == "__main__":
    #sov = compare_omega(omega)
    #plot_omega(sov)
    #jacobi_data, gauss_data, sov_data = convergence_iteration()
    #plot_convergence_delta(jacobi_data, gauss_data, sov_data)
    #over_relaxation_variable()
    compare_to_analytical()
