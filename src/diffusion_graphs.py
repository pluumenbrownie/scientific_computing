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
threshold = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
omega = np.linspace(0.5,1.95,10) # typical value for omega


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
    analytical = analytical_solution(y_values, 1)
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

def convergence_iteration ():
    jacobi_list = []
    gauss_list = []
    sov_list = []

    for th in threshold:
        print("start simulation for jacobi")
        jacobi = Jacobi(N=N, threshold = th)
        iteration = jacobi.run()
        jacobi_list.append(iteration)
    
    for th in threshold:
        print("start simualtion for GaussSeidel")
        gauss = GaussSeidel(N=N, threshold = th)
        iteration = gauss.run()
        gauss_list.append(iteration)
    
    for th in threshold:
        print("start simulation for SOV")
        sov = SuccessiveOverRelaxation(omega = 1.8, N=N, threshold = th)
        iteration = sov.run()
        sov_list.append(iteration)
    
    print("finish simulation")
    return jacobi_list, gauss_list, sov_list

def compare_omega(omega):
    sov_list = []
    for o in omega:
        sov = SuccessiveOverRelaxation(omega = o, N=N, threshold = 1e-5)
        iteration = sov.run()
        sov_list.append(iteration)
    return sov_list


def plot_convergence_ite (jacobi, gauss, sov):
    threshold_log = -np.log10(threshold) # convert to log scale
    plt.plot(threshold_log, jacobi, "o-", color = "red", label = "Jacobi")
    plt.plot(threshold_log, gauss, "o-", color = "green", label = "Gauss-Seidel")
    plt.plot(threshold_log, sov, "o-", color = "blue", label = "SOV")
    plt.legend()
    plt.title("Threshold value vs. Iteration")
    plt.xlabel("P (threshold value = 10^(-P))")
    plt.ylabel("Iterations")
    plt.grid()
    plt.savefig("local/threshold_iterations.png", dpi=300)
    plt.show()


def plot_omega (sov):
    plt.plot(omega, sov, 'o-')
    plt.xlabel("omega value")
    plt.ylabel("iteration")
    plt.title("SOV iteration with different omega values")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # compare_to_analytical()
    #jacobi, gauss, sov = convergence_iteration()
    #plot_convergence_ite(jacobi,gauss, sov)
    sov = compare_omega(omega)
    plot_omega(sov)
