import numpy as np
import os
import matplotlib.pyplot as plt


class Leapfrog:
    """
    Implement the leapfrog method for a 1D harmonic oscillator under
    the Hookes Law: F(x) = -kx;
    Position is updated for each time step, while velocity is updated for
    each half time step

    """

    def __init__(
        self,
        mass: float,
        k: float,
        simulation: int,
        time_step: float,
        A: float = 1.0,
        o: float = 1.0,
    ):
        self.time_step = time_step
        self.simulation = simulation  # number of time steps
        self.mass = mass  # mass
        self.k = k  # spring constant
        self.A = A  # amplitude of the driving force
        self.omega = o  # frequency of the driving force
        self.total_time = self.simulation * self.time_step
        self.position = np.zeros(self.simulation)  # store the value of position
        self.velocity = np.zeros(self.simulation)  # store the value of velocity

    def initial_condition(self):
        """
        Set the initial condition for the simulaton
        """
        self.position[0] = 1.0  # start from 1.0
        self.velocity[0] = 0.0

    def leap_frog(self):
        """
        Simulate the leap frog calculation, based on Hooke's law
        """
        self.initial_condition()

        # initialize v_half
        v_half = 0.5 * self.k * self.position[0] * self.time_step / self.mass
        self.velocity[0] = v_half

        for n in range(1, self.simulation):
            # update position at full-step using v at half-step
            self.position[n] = self.position[n - 1] + v_half * self.time_step
            F_new = -self.k * self.position[n]

            # update velocity
            v_half += (F_new / self.mass) * self.time_step

            # store velocity at half step
            self.velocity[n] = v_half

    def time_dependent_force(self, time: float):
        """
        Add a time dependent sinusoidal driving force to the 1D osillator
        """
        self.initial_condition()

        # initialize v_half
        v_half = 0.5 * self.k * self.position[0] * self.time_step / self.mass
        self.velocity[0] = v_half

        for n in range(1, self.simulation):
            time = n * self.time_step
            self.position[n] = self.position[n - 1] + v_half * self.time_step
            F_new = -self.k * self.position[n] + self.A * np.sin(self.omega * time)

            # update the velocity for half time step
            v_half += F_new / self.mass * self.time_step

            # store velocity at half step
            self.velocity[n] = v_half

    def plot(self, name: str):
        """
        Plot the the result for leap-frog simulation
        """
        plt.plot(
            np.linspace(0, self.total_time, self.simulation),
            self.position,
            label="Position (x)",
        )
        plt.plot(
            np.linspace(0, self.total_time, self.simulation),
            self.velocity,
            label="Velocity (v)",
        )
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.title(f"Leap-Frog Integration of Oscillatory Motion, {name}", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        savepath = "./figures"
        os.makedirs(savepath, exist_ok=True)
        filepath = os.path.join(savepath, f"leap frog {name} .png")
        plt.savefig(filepath, dpi=300)
        plt.show()
        plt.close()

    def phase_plot(self, name: str):
        """
        Plot the phase plot (v,x) for leap-frog simulation with
        time-dependent sinusoidal force
        """
        # plot the spiral
        plt.plot(self.position, self.velocity, label="Phase trajectory")
        plt.plot(self.position[0], self.velocity[0], "ro", label="start point")
        plt.plot(self.position[-1], self.velocity[-1], "go", label="end point")
        plt.xlabel("Position(x)", fontsize=14)
        plt.ylabel("Velocity(v)", fontsize=14)
        plt.title(
            f"Phase plot of 1D oscillator with time-dependent force, $\\omega = ${self.omega}",
            fontsize=14,
        )
        plt.grid()
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        savepath = "./figures"
        os.makedirs(savepath, exist_ok=True)
        filepath = os.path.join(savepath, f"leap frog {name} .png")
        plt.savefig(filepath, dpi=300)
        plt.show()
        plt.close()


if __name__ == "__main__":
    lf = Leapfrog(mass=1.0, k=1.0, simulation=1000, time_step=0.01)
    lf.leap_frog()
    lf.plot("k=1.0")

    lf2 = Leapfrog(mass=1.0, k=3.0, simulation=1000, time_step=0.01)
    lf2.leap_frog()
    lf2.plot("k=3.0")

    lf3 = Leapfrog(mass=1.0, k=5.0, simulation=1000, time_step=0.01)
    lf3.leap_frog()
    lf3.plot("k=5.0")

    # phase plots with time-dependent force
    lf4 = Leapfrog(mass=1.0, k=4.0, simulation=2000, time_step=0.01, A=1.0, o=1.5)
    lf4.time_dependent_force(time=0.01)
    lf4.phase_plot("k = 4.0, omega = 1.5")

    lf5 = Leapfrog(mass=1.0, k=4.0, simulation=2000, time_step=0.01, A=1.0, o=2.0)
    lf5.time_dependent_force(time=0.01)
    lf5.phase_plot("k = 4.0, omega = 2.0")
