# Monte Carlo simulation of DLA
import taichi as ti  # type: ignore
import numpy as np

ti.init(arch=ti.cpu)


@ti.data_oriented
class monte_carlo:

    def __init__(
        self, N: int, particle: int, eta: float = 1, stick_p: float = 1
    ) -> None:
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
        for _ in range(10):  # run 10 walkers at the same time
            if self.particles_added[None] >= self.particle:
                continue  # stop adding new walkers

            x = ti.random(int) % self.N
            y = self.N - 1
            stuck = False  # flag

            while not stuck:
                dx, dy = (
                    ti.random(int) % 3 - 1,
                    ti.random(int) % 3 - 1,
                )  # randomly select a neighbour
                x = self.periodic_boundary(x + dx)  # update position of x and y
                y += dy

                if (y < 0) or (y >= self.N):
                    break  # reset the walker

                for nx in range(x - 1, x + 2):
                    for ny in range(y - 1, y + 2):
                        if (0 <= ny < self.N) and self.grid[
                            self.periodic_boundary(nx), ny
                        ] == 1:
                            self.grid[x, y] = 1  # grow the cluster
                            stuck = True
                            self.particles_added[None] += 1
                            break  # move to the next walker

                        if stuck:
                            break

    def gui_visual(self, scale: int = 5):
        """
        Visuslize the DLA process with GUI
        """
        res = scale * self.N
        self.grid_initialize()
        self.gui = ti.GUI("Monte Carlo DLA", res=(res, res))
        self.particles_added[None] = 0

        while self.gui.running:
            self.random_walk()  # run the simulation
            img = np.array(self.grid.to_numpy(), dtype=float)
            img = img[::-1, :]  # flip to the correct direction

            img_resized = np.zeros(shape=(res, res, 3))
            for i, j in np.ndindex(res, res):
                img_resized[i, j] = img[i // scale, j // scale]  # resize the image

            self.gui.set_image(img_resized)
            self.gui.show()


mdla = monte_carlo(N=50, particle=1000)
mdla.gui_visual()
