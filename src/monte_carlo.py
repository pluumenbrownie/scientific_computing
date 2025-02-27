# Monte Carlo simulation of DLA
import taichi as ti  # type: ignore
import numpy as np
from numpy.typing import NDArray
import os
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)


@ti.data_oriented
class monte_carlo:

    def __init__(self, N: int, particle: int, stick_p: float, eta: float = 1) -> None:
        self.N = N  # grid size
        self.particle = particle
        self.grid = ti.field(dtype=float, shape=(self.N, self.N))
        self.eta = eta  # classic DLA
        self.stick_p = stick_p  # sticking probability
        self.particles_added = ti.field(dtype=float, shape=())

    @ti.kernel
    def grid_initialize(self):
        """
        Initialize the grid and seed for simulation
        """
        for i, j in ti.ndrange(self.N, self.N):
            self.grid[i, j] = 0  # intialize the empty grid
        self.grid[self.N // 2, 0] = 1  # initialize the seed at the bottom boundary

    @ti.func
    def periodic_boundary(self, i: int) -> int:
        """
        Apply periodic boundary to the grid
        """
        return i % self.N

    @ti.kernel
    def random_walk(self):
        """
        Particles randomly walk in the space. Start from the top boundary.
        Remove ones that move out of the boundary and create new ones
        """
        for _ in range(10):  # run 5 walkers at the same time
            if self.particles_added[None] >= self.particle:
                continue  # stop adding new walkers

            x = ti.random(int) % self.N
            y = self.N - 1
            stuck = False  # flag

            while not stuck:
                dx, dy = (
                    ti.random(int) % 3 - 1,
                    ti.random(int) % 3 - 1,
                )  # randomly select a neighbouring grid to walk

                da = self.periodic_boundary(x + dx)
                if (0 <= y + dy < self.N) and (self.grid[da, y + dy] == 0):
                    x = da  # update position of x and y only when the spot is empty
                    y += dy

                if (y < 0) or (y >= self.N):
                    break  # reset the walker

                for nx in range(x - 1, x + 2):
                    for ny in range(y - 1, y + 2):
                        if (0 <= ny < self.N) and self.grid[
                            self.periodic_boundary(nx), ny
                        ] == 1:
                            p = ti.random(ti.f32)
                            if p < self.stick_p:
                                self.grid[x, y] = (
                                    1  # grow the cluster only for sticking probability
                                )
                                stuck = True
                                self.particles_added[None] += 1
                            if self.particles_added[None] == self.particle:
                                print("Simulation finished")  # stop adding new workers
                            if stuck:
                                break  # move to the next walker

                    if stuck:
                        break  # move to the next walker

    def gui_visual(self, name: str, scale: int = 5):
        """
        Visuslize the DLA process with GUI
        """
        res = scale * self.N
        self.grid_initialize()
        self.gui = ti.GUI("Monte Carlo DLA", res=(res, res))
        self.particles_added[None] = 0

        while self.gui.running:
            if (
                self.particles_added[None] >= self.particle
            ):  # Stop simulation when complete
                break

            self.random_walk()  # run the simulation
            img = np.array(self.grid.to_numpy(), dtype=float)
            img = img[::-1, :]  # flip to the correct direction

            img_resized = np.zeros(shape=(res, res, 3))
            for i, j in np.ndindex(res, res):
                img_resized[i, j] = img[i // scale, j // scale]  # resize the image

            self.gui.set_image(img_resized)
            self.gui.show()

        gray_img = np.mean(img_resized, axis=-1)
        self.save_gui(gray_img, name)
        print(
            f"Simulation complete. Image saved as {name}.jpg"
        )  # save the image after simulation

    def save_gui(self, image: NDArray, name: str):
        """
        This function saves the resulted cluster of Monte Carlo
        as a jpg file.
        """
        img = image.astype(np.uint8)
        img = np.rot90(img, k=1)
        save_folder = "local/"
        plt.imsave(os.path.join(save_folder, f"{name}.jpg"), img, cmap="gray", dpi=300)


# test with different sticking probability
mdla = monte_carlo(N=100, particle=1000, stick_p=0.1)
mdla.gui_visual(name="Monte Carlo_p_0.1_N_100")

mdla2 = monte_carlo(N=100, particle=1000, stick_p=0.5)
mdla2.gui_visual(name="Monte Carlo_p_0.5_N_100")

mdla3 = monte_carlo(N=100, particle=1000, stick_p=0.9)
mdla3.gui_visual(name="Monte Carlo_p_0.9_N_100")
