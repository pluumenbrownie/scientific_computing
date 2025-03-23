import taichi as ti
from collage_eigenmodes import *
from direct_method import direct_method
from leapfrog import Leapfrog


if __name__ == "__main__":

    # plot the colloge and frequency for question 3.1
    ti.init()

    eigenmode_collage("figures/eigenmode_collage.pdf")
    eigenfrequency_plot("figures/eigenfrequencies.pdf")
    animate_eigenvectors("figures")

    # plot the steady state concentration for question 3.2
    dm = direct_method(N=100)
    dm.matrix_construct()
    dm.matrix_solve()
    dm.plot()

    # plot the leapfrog integration for question 3.3
    lf = Leapfrog(mass=1.0, k=1.0, simulation=1000, time_step=0.01)
    lf.leap_frog()
    lf.plot("k=1.0")

    lf2 = Leapfrog(mass=1.0, k=3.0, simulation=1000, time_step=0.01)
    lf2.leap_frog()
    lf2.plot("k=3.0")

    lf3 = Leapfrog(mass=1.0, k=5.0, simulation=1000, time_step=0.01)
    lf3.leap_frog()
    lf3.plot("k=5.0")

    lf4 = Leapfrog(mass=1.0, k=4.0, simulation=2000, time_step=0.01, A=1.0, o=1.5)
    lf4.time_dependent_force(time=0.01)
    lf4.phase_plot("k = 4.0, omega = 1.5")

    lf5 = Leapfrog(mass=1.0, k=4.0, simulation=2000, time_step=0.01, A=1.0, o=2.0)
    lf5.time_dependent_force(time=0.01)
    lf5.phase_plot("k = 4.0, omega = 2.0")

    benchmark_solvers()
