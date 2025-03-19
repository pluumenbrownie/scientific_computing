import numpy as np
import matplotlib.pyplot as plt


class Leapfrog:
    """
    Implement the leapfrog method for a 1D harmonic oscillator under
    the Hookes Law: F(x) = -kx;
    Position is updated for each time step, while velocity is updated for
    each half time step

    """

    def __init__(self, mass: float, k: float, simulation: int, time_step: float):
        self.time_step = time_step
        self.simulation = simulation  # number of time steps
        self.mass = mass  # mass
        self.k = k  # spring constant
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
        v_half = 0
        for n in range(1, self.simulation):
            F = -self.k * self.position[n - 1]  # force at current position
            v_half += (
                F / self.mass * self.time_step
            )  # update velocity for half time step
            self.position[n] = (
                self.position[n - 1] + v_half * self.time_step
            )  # update position using velocity at half time step

            F_new = -self.k * self.position[n]
            v_half += (F_new / self.mass) * self.time_step  # update the velocity again
            self.velocity[n] = v_half

    def plot(self):
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
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("Leap-Frog Integration of Oscillatory Motion")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    lf = Leapfrog(mass=1.0, k=1.0, simulation=1000, time_step=0.01)
    lf.leap_frog()
    lf.plot()

    lf2 = Leapfrog(mass=1.0, k=3.0, simulation=1000, time_step=0.01)
    lf2.leap_frog()
    lf2.plot()

    lf3 = Leapfrog(mass=1.0, k=5.0, simulation=1000, time_step=0.01)
    lf3.leap_frog()
    lf3.plot()
