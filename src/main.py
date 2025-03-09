from gray_scott_experiments import pyplot_combined
from numpy.random import seed
from diffusion_limited import simulate_dla, plot_concentration_and_dla
from monte_carlo import monte_carlo

if __name__ == "__main__":

    # Fix numpy seed to prevent unnessecary figure updates
    seed(12650662)

    # Gray-Scott collage
    pyplot_combined(save_location="./figures/gray_scott_collage.pdf")

    # run the following code for DLA simulation with Monte Carlo Simulation
    mdla = monte_carlo(N=100, particle=1000, stick_p=0.1)
    mdla.gui_visual(name="Monte Carlo_p_0.1_N_100")

    mdla2 = monte_carlo(N=100, particle=1000, stick_p=0.5)
    mdla2.gui_visual(name="Monte Carlo_p_0.5_N_100")

    mdla3 = monte_carlo(N=100, particle=1000, stick_p=0.9)
    mdla3.gui_visual(name="Monte Carlo_p_0.9_N_100")

    # run the following code for DLA simulation with SOR iteration
    eta_l = [0.6, 1.0, 1.2]
    for eta in eta_l:
        simulate_dla(eta)
        plot_concentration_and_dla(eta)
