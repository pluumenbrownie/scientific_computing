import taichi as ti
import numpy as np
from numpy.typing import NDArray
from typing import Any, Self
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


@ti.data_oriented
class Membrane:
    """
    A model for a vibrating membrane or drum.

    Please create via `Membrane.square()`, `Membrane.rectangle()`, or
    `Membrane.circle()`, not via direct assignment.

    - `membrane`: a field representing the deflection of a cell
    - `cell_number`: cells are numbered sequentially. If zero, the cell is boundary.
    - `cell_count`: the amount of cells.
    """

    membrane: ti.ScalarField | ti.MatrixField
    cell_number: ti.ScalarField | ti.MatrixField
    cell_count: int

    def __init__(self, membrane: Any) -> None:
        self.membrane = membrane
        self.cell_number = ti.field(shape=self.membrane.shape, dtype=int)
        # self.cell_count = ti.field(shape=(), dtype=int)

    @classmethod
    def square(cls, L: int) -> Self:
        """
        Returns a square membrane.
        """
        membrane = cls(ti.field(shape=(L, L), dtype=float))
        membrane.cell_count = membrane.number_cells()
        return membrane

    @classmethod
    def rectangle(cls, L1: int, L2: int) -> Self:
        """
        Returns a rectangular membrane.
        """
        membrane = cls(ti.field(shape=(L1, L2), dtype=float))
        membrane.cell_count = membrane.number_cells()
        return membrane

    @classmethod
    def circle(cls, L: int) -> Self:
        """
        Returns a circular membrane.
        """
        membrane = cls(ti.field(shape=(L, L), dtype=float))
        membrane.cell_count = membrane.number_circle()
        return membrane

    @ti.kernel
    def number_cells(self) -> int:
        cell_count = 0
        ti.loop_config(serialize=True)
        for i, j in self.membrane:
            cell_count += 1
            self.cell_number[i, j] = cell_count
        return cell_count

    @ti.kernel
    def number_circle(self) -> int:
        MID = (self.membrane.shape[0] - 1) / 2.0
        cell_count = 0
        ti.loop_config(serialize=True)
        for i, j in self.cell_number:
            if (i - MID) ** 2 + (j - MID) ** 2 >= (MID**2 + 1):
                self.cell_number[i, j] = 0
            else:
                cell_count += 1
                self.cell_number[i, j] = cell_count
        return cell_count

    @ti.kernel
    def draw(self, scale: int):
        for i, j in self.image:
            if self.cell_number[i // scale, j // scale] == 0:
                self.image[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self.image[i, j] = (
                    self.membrane[i // scale, j // scale] / 2.0 + 0.5
                ) * ti.Vector([1.0, 1.0, 1.0])

    def show(self, scale: int = 1):
        """
        Run diffusion steps and show the resulting diffusions live until closed.

        # Inputs:
        - scale: How much the shown video should be scaled. Default = 1
        """
        i_size, j_size = self.membrane.shape
        scaled_size = (scale * i_size, scale * j_size)
        gui = ti.GUI("Membrane example", res=scaled_size, fast_gui=True)
        self.image = ti.Vector.field(3, float, shape=scaled_size)

        while gui.running:
            self.draw(scale)
            gui.set_image(self.image)
            gui.show()


@ti.data_oriented
class Solver:
    def __init__(self, membrane: Membrane) -> None:
        self.adjecency_matrix = np.zeros(
            shape=(membrane.cell_count, membrane.cell_count)
        )
        self.ranked_membrane = membrane.cell_number.to_numpy()
        for i, j in np.ndindex(self.ranked_membrane.shape):
            if self.ranked_membrane[i, j] == 0:
                continue
            cell_rank = self.ranked_membrane[i, j] - 1
            self.adjecency_matrix[cell_rank, cell_rank] = -4.0
            for ni, nj in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:
                if (
                    ni < 0
                    or ni >= self.ranked_membrane.shape[0]
                    or nj < 0
                    or nj >= self.ranked_membrane.shape[1]
                    or self.ranked_membrane[ni, nj] == 0
                ):
                    continue
                neighbor_rank = self.ranked_membrane[ni, nj] - 1
                self.adjecency_matrix[cell_rank, neighbor_rank] = 1.0

    def solve(self):
        print(eigh(self.adjecency_matrix)[0])


@ti.data_oriented
class TaichiSolver:
    def __init__(self, membrane: Membrane) -> None:
        self.adjecency_matrix = ti.field(
            ti.f64, shape=(membrane.cell_count, membrane.cell_count)
        )
        self.ranked_membrane = membrane.cell_number
        self.construct_adjecency_matrix()

    @ti.kernel
    def construct_adjecency_matrix(self):
        for i, j in self.ranked_membrane:
            if self.ranked_membrane[i, j] == 0:
                continue
            cell_rank = self.ranked_membrane[i, j] - 1
            self.adjecency_matrix[cell_rank, cell_rank] = -4.0
            for ni, nj in ti.static([[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]):
                if not (
                    ni < 0
                    or ni >= self.ranked_membrane.shape[0]
                    or nj < 0
                    or nj >= self.ranked_membrane.shape[1]
                    or self.ranked_membrane[ni, nj] == 0
                ):
                    neighbor_rank = self.ranked_membrane[ni, nj] - 1
                    self.adjecency_matrix[cell_rank, neighbor_rank] = 1.0

    def solve(self):
        print(eigh(self.adjecency_matrix.to_numpy())[0])


@ti.data_oriented
class SparseSolver(TaichiSolver):
    def solve(self):
        """
        Need to fine tune `k` to get the desired amount of eigenvalues.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh
        https://stackoverflow.com/questions/11083660/python-eigenvectors-differences-among-numpy-linalg-scipy-linalg-and-scipy-spar?rq=3
        https://en.wikipedia.org/wiki/Lanczos_algorithm
        """
        print(eigsh(self.adjecency_matrix.to_numpy(), k=15)[0])


if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    mem = Membrane.square(4)
    print(f"{mem.cell_count = }")
    solver = Solver(mem)
    # print(solver.adjecency_matrix)
    solver.solve()
    tisolver = TaichiSolver(mem)
    # print(tisolver.adjecency_matrix)
    tisolver.solve()
    spsolver = SparseSolver(mem)
    # print(spsolver.adjecency_matrix)
    spsolver.solve()
    # mem.show(scale=4)
