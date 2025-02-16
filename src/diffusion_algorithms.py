from typing import Callable
import numpy as np
import taichi as ti
import math as mt


ti.init(arch=ti.cpu)  # change this if you have gpu


# Parameters
DEFAULT_N = 50  # gridpoints
DEFAULT_OMEGA = 1.8  # relaxation constant
THRESHOLD = 1e-5


@ti.data_oriented
class BaseIteration:
    """Do not use."""

    def __init__(self, N: int = DEFAULT_N, threshold: float = THRESHOLD) -> None:
        self.threshold = threshold
        self.N = N
        self.concentration = ti.field(float, shape=(N, N))
        self.c_difference = ti.field(float, shape=(N, N))
        self.init()

    def run(self) -> int:
        """
        Run this solving algorithm until the changes are smaller then `self.threshold`.
        """
        self.solve()
        runs = 1
        while self.c_difference.to_numpy().max() > self.threshold:
            self.solve()
            runs += 1
        return runs

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
    def neighbourhood_values(
        self, i: int, j: int, field: ti.ScalarField | ti.MatrixField
    ) -> float:
        """
        Get the combined values of the four neighbours of the given cell,
        using boundary conditions.
        """
        return (
            field[self.pbi(i - 1), j]
            + field[self.pbi(i + 1), j]
            + field[self.pbi(i), j - 1]
            + field[self.pbi(i), j + 1]
        )

    @ti.kernel
    def init_concentration(self):
        """
        Reset `self.concentration` back to the inital conditions.
        """
        for i, j in self.concentration:
            if j == 0:
                self.concentration[i, j] = 0
            elif j == self.N - 1:
                self.concentration[i, j] = 1
            else:
                self.concentration[i, j] = 0

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
            self.c_difference[i, j] = abs(
                self.concentration[i, j] - self.c_difference[i, j]
            )

    @ti.func
    def reset_boundary(self):
        """
        Enforce the values of the boundary conditions.
        """
        for i in range(self.N):
            self.concentration[i, 0] = 1.0
            self.concentration[i, self.N - 1] = 0.0

    def gui(self, scale: int = 10):
        """
        Show the state of `self.concentration`.

        Window can be made bigger or smaller with `scale`.
        """
        gui = ti.GUI(
            f"Diffusion {self.N}x{self.N}",
            res=(scale * self.N, scale * self.N),  # type:ignore
        )
        image = self.concentration.to_numpy()

        # scale image if necessary
        if not scale == 1:
            scaled_image = np.zeros(shape=(scale * self.N, scale * self.N))
            for i, j in np.ndindex(scaled_image.shape):
                scaled_image[i, j] = image[i // scale, j // scale]
            image = scaled_image

        while gui.running:
            gui.set_image(image)
            gui.show()


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
            if j == 0 or j >= self.N - 1:
                continue
            self.concentration[i, j] = 0.25 * self.neighbourhood_values(
                i, j, self.c_difference
            )
        self.reset_boundary()
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
            if j == 0 or j >= self.N - 1:
                continue
            self.concentration[i, j] = 0.25 * self.neighbourhood_values(
                i, j, self.concentration
            )

        for v in self.white_tiles:
            i, j = int(self.white_tiles[v][0]), int(self.white_tiles[v][1])
            if j == 0 or j >= self.N - 1:
                continue
            self.concentration[i, j] = 0.25 * self.neighbourhood_values(
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
            if j == 0 or j >= self.N - 1:
                continue
            self.concentration[i, j] = (
                self.omega * 0.25 * self.neighbourhood_values(i, j, self.concentration)
                + (1 - self.omega) * self.concentration[i, j]
            )

        for v in self.white_tiles:
            i, j = int(self.white_tiles[v][0]), int(self.white_tiles[v][1])
            if j == 0 or j >= self.N - 1:
                continue
            self.concentration[i, j] = (
                self.omega * 0.25 * self.neighbourhood_values(i, j, self.concentration)
                + (1 - self.omega) * self.concentration[i, j]
            )
        self.calculate_differences()


if __name__ == "__main__":
    sov = SuccessiveOverRelaxation()
    sov_amount = sov.run()
    print(f"{sov_amount = }")
    sov.gui()
