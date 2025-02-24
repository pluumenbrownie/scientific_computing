from typing import Callable
import numpy as np
import taichi as ti
import math as mt
from cell_type import CellTypes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os


ti.init(arch=ti.cpu)  # change this if you have gpu


# Parameters
DEFAULT_N = 50  # gridpoints
DEFAULT_OMEGA = 1.8  # relaxation constant
THRESHOLD = 1e-5
Vec2 = ti.types.vector(2, float)


@ti.data_oriented
class BaseIteration:
    """Do not use."""

    def __init__(self, N: int = DEFAULT_N, threshold: float = THRESHOLD) -> None:
        self.threshold = threshold
        self.N = N
        self.concentration = ti.Vector.field(n=2, dtype=float, shape=(N, N))
        self.c_difference = ti.Vector.field(n=2, dtype=float, shape=(N, N))
        self.init()

    def run(self) -> tuple[int, list[float]]:
        """
        Run this solving algorithm until the changes are smaller then `self.threshold`
        """
        self.solve()
        runs = 1
        delta_values = [self.c_difference.to_numpy()[:, :, 0].max()]

        while delta_values[-1] > self.threshold:
            self.solve()
            runs += 1
            delta_values.append(
                self.c_difference.to_numpy()[:, :, 0].max()
            )  # Store delta for plotting

        return runs, delta_values

    def add_rectangle(self, x1, x2, y1, y2):
        """Marks a rectangular region."""
        for i in range(x1, x2):
            for j in range(y1, y2):
                self.concentration[i, j][1] = CellTypes.sink  # type: ignore

    def add_insulator(self, x1, x2, y1, y2):
        """Marks a rectangular region."""
        for i in range(x1, x2):
            for j in range(y1, y2):
                self.concentration[i, j][1] = CellTypes.blocker  # type: ignore

    def init(self):
        """
        Fill all of the fields in `self` with the required initial values.
        """
        self.init_concentration()

    @ti.kernel
    def solve(self):
        """
        Perform a solving step on `self.concentration`.
        """
        raise NotImplementedError

    @ti.func
    def pbi(self, i: int) -> int:
        """
        Convert normal `i` coordinates into periodic boundary coordinates.
        """
        if i < 0:
            i += self.N
        if i >= self.N:
            i -= self.N
        return i

    @ti.func
    def average_neighbourhood_values(self, i: int, j: int, field) -> float:
        """
        Get the combined values of the four neighbours of the given cell,
        using boundary conditions.
        """
        # neighbour_count = self.neighbour_nonblocking_count(i, j, field)
        # assert neighbour_count == 4.0
        return (
            field[self.pbi(i - 1), j][0]
            + field[self.pbi(i + 1), j][0]
            + field[self.pbi(i), j - 1][0]
            + field[self.pbi(i), j + 1][0]
        ) / self.neighbour_nonblocking_count(i, j, field)

    @ti.func
    def neighbour_nonblocking_count(self, i: int, j: int, field) -> float:
        return 4 - (
            int(field[self.pbi(i - 1), j][1] == CellTypes.blocker)  # type: ignore
            + int(field[self.pbi(i + 1), j][1] == CellTypes.blocker)  # type: ignore
            + int(field[self.pbi(i), j - 1][1] == CellTypes.blocker)  # type: ignore
            + int(field[self.pbi(i), j + 1][1] == CellTypes.blocker)  # type: ignore
        )

    @ti.func
    def static_cell(self, i: int, j: int):
        """
        Returns true if given cell should not be updated in the main update loop.
        """
        return (
            j == 0
            or j >= self.N - 1
            or not self.concentration[i, j][1] == CellTypes.normal  # type: ignore
        )

    @ti.kernel
    def init_concentration(self):
        """
        Reset `self.concentration` back to the inital conditions.
        """
        for i, j in self.concentration:
            if j == 0:
                self.concentration[i, j] = Vec2(0.0, CellTypes.sink)  # type: ignore
            elif j == self.N - 1:
                self.concentration[i, j] = Vec2(1.0, CellTypes.source)  # type: ignore
            else:
                self.concentration[i, j] = Vec2(0.0, CellTypes.normal)  # type: ignore

    @ti.func
    def copy_into_difference(self):
        """
        Copy the values of `self.concentration` into `self.c_difference`.
        """
        for i, j in self.concentration:
            self.c_difference[i, j] = self.concentration[i, j]

    @ti.func
    def calculate_differences(self):
        """
        Calculate the magnitude of the changes made in this step.

        Sets the values in `self.c_difference` to the absolute difference
        between the current values of `self.c_difference` and
        the values of `self.concentration.` Must run `self.copy_into_difference()`
        before changes are made to `self.concentration` to be usefull.
        """
        for i, j in self.c_difference:
            self.c_difference[i, j][0] = abs(
                self.concentration[i, j][0] - self.c_difference[i, j][0]
            )

    @ti.func
    def reset_boundary(self):
        """
        Enforce the values of the boundary conditions.
        """
        for i in range(self.N):
            self.concentration[i, 0][0] = 0.0
            self.concentration[i, self.N - 1][0] = 1.0

    def gui(self, name, scale: int = 10):
        """
        Show the state of `self.concentration`.

        Window can be made bigger or smaller with `scale`.
        """
        gui = ti.GUI(
            f"Diffusion {self.N}x{self.N}",
            res=(scale * self.N, scale * self.N),  # type:ignore
        )
        # self.concentration.to_numpy().shape = (N, N, 2)
        image = self.concentration.to_numpy()[:, :, 0]

        # add colors to image
        norm = mcolors.Normalize(vmin=image.min(), vmax=image.max())
        colored_image = cm.viridis(norm(image))[:, :, :3]  # type: ignore

        # scale image if necessary
        if not scale == 1:
            scaled_image = np.zeros(shape=(scale * self.N, scale * self.N, 3))
            for i, j in np.ndindex(scale * self.N, scale * self.N):
                scaled_image[i, j] = colored_image[i // scale, j // scale]
            colored_image = scaled_image

        while gui.running:
            gui.set_image(colored_image)
            gui.show()

        img_array = (colored_image * 255).astype(np.uint8)
        img_array = np.rot90(img_array, k=1)
        save_folder = "local/"
        plt.imsave(os.path.join(save_folder, f"{name}.jpg"), img_array, dpi=300)


class Jacobi(BaseIteration):
    """
    Solve the diffusion equation by taking the neighbouring values from a copy
    of the previous state.

    # Inputs
    - `threshold`: The minimal amount change needed for `self.run()` to keep
    iterating. Default `threshold = 1e-5`
    - `N`: The size of the grid. Default `N = 50`
    """

    @ti.kernel
    def solve(self):
        self.copy_into_difference()
        for i, j in self.concentration:
            if self.static_cell(i, j):
                continue
            self.concentration[i, j][0] = self.average_neighbourhood_values(
                i, j, self.c_difference
            )
        self.calculate_differences()


class GaussSeidel(BaseIteration):
    """
    Solve the diffusion equation by taking the neighbouring values from the
    current state in place. Uses a checkerboard pattern to prevent race
    conditions.

    # Inputs
    - `threshold`: The minimal amount change needed for `self.run()` to keep
    iterating. Default `threshold = 1e-5`
    - `N`: The size of the grid. Default `N = 50`
    """

    def __init__(self, N: int = DEFAULT_N, threshold: float = THRESHOLD) -> None:
        super().__init__(N, threshold)
        self.white_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.ceil(N**2 / 2)))
        self.black_tiles = ti.Vector.field(n=2, dtype=int, shape=(mt.floor(N**2 / 2)))
        self.init_checkerboard()

    @ti.kernel
    def init_checkerboard(self):
        """
        Fills the checkerboard fields `self.white_tiles` and `self.black_tiles`.
        """
        for i, j in self.concentration:
            if i % 2 == j % 2:
                self.white_tiles[j // 2 + ti.ceil(i * self.N / 2, dtype=int)] = (
                    ti.Vector([i, j])
                )
            else:
                self.black_tiles[j // 2 + ti.floor(i * self.N / 2, dtype=int)] = (
                    ti.Vector([i, j])
                )

    @ti.kernel
    def solve(self):
        self.copy_into_difference()
        for v in self.black_tiles:
            i, j = int(self.black_tiles[v][0]), int(self.black_tiles[v][1])
            if self.static_cell(i, j):
                continue
            self.concentration[i, j][0] = self.average_neighbourhood_values(
                i, j, self.concentration
            )

        for v in self.white_tiles:
            i, j = int(self.white_tiles[v][0]), int(self.white_tiles[v][1])
            if self.static_cell(i, j):
                continue
            self.concentration[i, j][0] = self.average_neighbourhood_values(
                i, j, self.concentration
            )
        self.calculate_differences()


class SuccessiveOverRelaxation(GaussSeidel):
    """
    Solve the diffusion equation by proportionally taking the neighbouring
    values from the current state and the current value in place. Uses a
    checkerboard pattern to prevent race conditions. For "over relaxation",
    the value `self.omega` can be made greater than `1,0`.

    # Inputs
    - `threshold`: The minimal amount change needed for `self.run()` to keep
    iterating. Default `threshold = 1e-5`
    - `N`: The size of the grid. Default `N = 50`
    - `omega`: The relaxation constant. Default `omega = 1.8`
    """

    def __init__(
        self,
        omega: float = DEFAULT_OMEGA,
        N: int = DEFAULT_N,
        threshold: float = THRESHOLD,
    ) -> None:
        super().__init__(N, threshold)
        self.omega = omega

    @ti.kernel
    def solve(self):
        self.copy_into_difference()
        for v in self.black_tiles:
            i, j = int(self.black_tiles[v][0]), int(self.black_tiles[v][1])
            if self.static_cell(i, j):
                continue
            self.concentration[i, j][0] = (
                self.omega * self.average_neighbourhood_values(i, j, self.concentration)
                + (1 - self.omega) * self.concentration[i, j][0]
            )

        for v in self.white_tiles:
            i, j = int(self.white_tiles[v][0]), int(self.white_tiles[v][1])
            if self.static_cell(i, j):
                continue
            self.concentration[i, j][0] = (
                self.omega * self.average_neighbourhood_values(i, j, self.concentration)
                + (1 - self.omega) * self.concentration[i, j][0]
            )
        self.reset_boundary()
        self.calculate_differences()


if __name__ == "__main__":
    # sov = SuccessiveOverRelaxation()
    # sov_amount = sov.run()
    # print(f"{sov_amount = }")
    # sov.gui()

    jacobe = SuccessiveOverRelaxation()
    jacobe.add_rectangle(30, 35, 30, 35)
    jacobe.add_rectangle(15, 20, 15, 20)
    jacobe.run()
    jacobe.gui(name="sinks")

    """N = 50  # Grid size
    jacobi = Jacobi(N=N)
    jacobi.add_insulator(5, 10, 5, 10)  # Add an obstacle
    jacobi.add_insulator(40, 45, 5, 10)
    jacobi.add_insulator(5, 10, 40, 45)
    jacobi.add_insulator(40, 45, 40, 45)
    jacobi.run()"""

    # gauss = GaussSeidel(N=N)
    # gauss.add_rectangle(10, 15, 10, 15)  # Add an obstacle
    # gauss.run()

    N = 50
    sor = SuccessiveOverRelaxation(omega=1.8, N=N)
    sor.add_insulator(30, 35, 30, 35)
    sor.add_insulator(15, 20, 15, 20)
    sor.run()
    sor.gui(name="insulators")

    # jacobi.gui()
    # gauss.gui()
    # sor.gui()
    # print(f"{runs = }")

    # sor = SuccessiveOverRelaxation(omega=1.8, N=N)
    # sor.add_rectangle(25, 35, 25, 35)  # Another an obstacle
    # runs, _ = sor.run()
    # sor.gui()
    # print(f"{runs = }")
