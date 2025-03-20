import taichi as ti
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)


@ti.data_oriented
class direct_method:
    """
    Solve the diffusion equation using the direct method
    matrix equation MC = b
    domain: circular disk with a radius = 2
    source: (0.6,1.2) with concentration = 1
    concentration = 0 outside the domain
    """

    def __init__(self, N: int, source_x: float = 0.6, source_y: float = 1.2):
        self.N = N  # grid size
        self.concentration = ti.field(dtype=float, shape=(self.N, self.N))
        self.L = 2.0  # domain radius
        self.dx = (2 * self.L) / self.N  # grid spacing
        self.dy = (2 * self.L) / self.N
        self.source_x = int(
            (source_x + self.L) / self.dx
        )  # convert coordinates into grid indices
        self.source_y = int((source_y + self.L) / self.dy)
        self.b = np.zeros(self.N**2)  # vectors on the right
        self.M = sp.lil_matrix((self.N**2, self.N**2))  # laplace matrix

    @ti.kernel
    def grid_initialize(self):
        """
        Initialize the grid with boundary and source
        """
        for i, j in ti.ndrange(self.N, self.N):
            self.concentration[i, j] = 0

        self.concentration[self.source_x, self.source_y] = 1.0  # source

    def matrix_construct(self):
        """
        Construct the Laplacian matrix M and the right-hand side vector b.
        """
        self.grid_initialize()

        for i in range(self.N):
            for j in range(self.N):
                x = -self.L + i * self.dx  # convert index to x-coordinate
                y = -self.L + j * self.dy
                index = j * self.N + i

                # apply boundary conditions
                if (x**2 + y**2) > self.L**2:
                    self.M[index, index] = 1
                    self.b[index] = 0
                    continue

                # apply source condition
                if i == self.source_x and j == self.source_y:
                    self.M[index, index] = 1
                    self.b[index] = 1
                    continue

                # apply 5-point stencil for Laplacian
                self.M[index, index] = -4
                if i > 0:  # Left
                    self.M[index, index - 1] = 1
                if i < self.N - 1:  # Right
                    self.M[index, index + 1] = 1
                if j > 0:  # Bottom
                    self.M[index, index - self.N] = 1
                if j < self.N - 1:  # Top
                    self.M[index, index + self.N] = 1

        # convert matrix to CSR format
        self.M = self.M.tocsr()

    def matrix_solve(self):
        """
        Solve the solution vector c
        """
        v = spla.spsolve(self.M, self.b)
        v_reshaped = v.reshape((self.N, self.N))

        self.copy_solution_to_taichi(v_reshaped)  # transform into tichi field

    @ti.kernel
    def copy_solution_to_taichi(self, v: ti.types.ndarray()):
        """
        Copy the computed solution from NumPy to the Taichi field.
        """
        for i, j in ti.ndrange(self.N, self.N):
            self.concentration[i, j] = v[i, j]

    def plot(self):
        """
        Plot the steady concentration with the initial state and the boundary
        """
        concentration = self.concentration.to_numpy().T
        x = np.linspace(-self.L, self.L, self.N)
        y = np.linspace(-self.L, self.L, self.N)
        x_rescale, y_rescale = np.meshgrid(
            x, y, indexing="ij"
        )  # rescale the x and y coordinates

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(
            x_rescale,
            y_rescale,
            concentration,
            shading="auto",
            cmap="inferno",
            vmin=0,
            vmax=concentration.max(),
        )
        boundary = plt.Circle(
            (0, 0),
            self.L,
            color="red",
            fill=False,
            linestyle="dashed",
            label="circular boundary",
        )
        plt.gca().add_patch(boundary)  # plot the boundary in the figure
        plt.colorbar(label="Concentration")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        plt.title("Steady State Concentration under Direct Method")
        plt.legend(loc="lower right")

        savepath = "./figures"
        os.makedirs(savepath, exist_ok=True)
        filepath = os.path.join(savepath, "steady_concentration.png")
        plt.savefig(filepath, dpi=300)
        plt.show()


if __name__ == "__main__":
    dm = direct_method(N=100)
    dm.matrix_construct()
    dm.matrix_solve()
    dm.plot()
