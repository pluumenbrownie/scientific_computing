import taichi as ti
import numpy as np
from numpy.typing import NDArray
from typing import Any, Self
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import os


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
    image: ti.ScalarField | ti.MatrixField
    cell_count: int
    name: str

    def __init__(self, membrane: Any, h: float, ui_scale: int) -> None:
        self.membrane = membrane
        self.h = h
        self.cell_number = ti.field(shape=self.membrane.shape, dtype=int)
        self.scale = ui_scale
        self.new_image(ui_scale)

    @classmethod
    def square(cls, L: float, h: float, ui_scale: int = 1) -> Self:
        """
        Returns a square membrane.

        :param L: The sidelength of the membrane.
        :type L: float
        :param h: The grid spacing.
        :type h: float
        :param ui_scale: How much to scale the image when shown.
        :type ui_scale: int
        :return:
        :rtype: Self
        """
        N = round(L / h)
        membrane = cls(ti.field(shape=(N, N), dtype=float), h, ui_scale=ui_scale)
        membrane.cell_count = membrane.number_cells()
        membrane.name = "Square"
        return membrane

    @classmethod
    def rectangle(cls, L1: float, L2: float, h: float, ui_scale: int = 1) -> Self:
        """
        Returns a rectangular membrane.

        :param L1: The horizontal sidelength of the membrane.
        :type L1: float
        :param L2: The vertial sidelength of the membrane.
        :type L2: float
        :param h: The grid spacing.
        :type h: float
        :param ui_scale: How much to scale the image when shown.
        :type ui_scale: int
        :return:
        :rtype: Self
        """
        N1 = round(L1 / h)
        N2 = round(L2 / h)
        membrane = cls(ti.field(shape=(N1, N2), dtype=float), h, ui_scale=ui_scale)
        membrane.cell_count = membrane.number_cells()
        membrane.name = "Rectangle"
        return membrane

    @classmethod
    def circle(cls, L: float, h: float, ui_scale: int = 1) -> Self:
        """
        Returns a circular membrane.

        :param L: The sidelength of the membrane.
        :type L: float
        :param h: The grid spacing.
        :type h: float
        :param ui_scale: How much to scale the image when shown.
        :type ui_scale: int
        :return:
        :rtype: Self
        """
        N = round(L / h)
        membrane = cls(ti.field(shape=(N, N), dtype=float), h, ui_scale=ui_scale)
        membrane.cell_count = membrane.number_circle()
        membrane.name = "Circle"
        return membrane

    @ti.kernel
    def number_cells(self) -> int:
        """
        Docstring for number_cells

        :return: The total amount of cells in the membrane.
        :rtype: int
        """
        cell_count = 0
        ti.loop_config(serialize=True)
        for i, j in self.membrane:
            cell_count += 1
            self.cell_number[i, j] = cell_count
        return cell_count

    @ti.kernel
    def number_circle(self) -> int:
        """
        Rank the cells which fall within the inscribed circle, set all others
        to 0.

        :return: The total amount of cells in the membrane.
        :rtype: int
        """
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
    def draw(self, abs_highest: float):
        """
        Draw an image to `self.image`.

        :param abs_highest: The highest absolute value in the dataset.
        :type abs_highest: float
        """
        for i, j in self.image:
            if self.cell_number[i // self.scale, j // self.scale] == 0:
                self.image[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                memb_value = self.membrane[i // self.scale, j // self.scale]
                if memb_value < 0:
                    self.image[i, j] = ti.Vector(
                        [
                            1.0 + memb_value / abs_highest,
                            1.0 + memb_value / abs_highest,
                            1.0,
                        ]
                    )
                else:
                    self.image[i, j] = ti.Vector(
                        [
                            1.0,
                            1.0 - memb_value / abs_highest,
                            1.0 - memb_value / abs_highest,
                        ]
                    )

    @ti.kernel
    def animated_draw(self, abs_highest: float, frequency: float, t: float):
        """
        Draw an image to `self.image` to use for an animation.

        :param abs_highest: The highest absolute value in the dataset.
        :type abs_highest: float
        :param frequency: The frequency to animate the oscillation at.
        :type frequency: float
        :param t: The point in time to draw.
        :type t: float
        """
        for i, j in self.image:
            if self.cell_number[i // self.scale, j // self.scale] == 0:
                self.image[i, j] = ti.Vector([0.0, 0.0, 0.0])
            else:
                memb_value = self.membrane[i // self.scale, j // self.scale]
                memb_value *= ti.sin(2 * np.pi * frequency * t)

                if memb_value < 0:
                    self.image[i, j] = ti.Vector(
                        [
                            1.0 + memb_value / abs_highest,
                            1.0 + memb_value / abs_highest,
                            1.0,
                        ]
                    )
                else:
                    self.image[i, j] = ti.Vector(
                        [
                            1.0,
                            1.0 - memb_value / abs_highest,
                            1.0 - memb_value / abs_highest,
                        ]
                    )

    def new_image(self, scale: int = 1) -> tuple[int, int]:
        """
        Create a `self.image` field and return its resolution.

        :param scale: The resulting image will be `scale * self.membrane.shape`.
        Default = 1
        :type scale: int
        :return: The resolution of the image.
        :rtype: tuple[int, int]
        """
        i_size, j_size = self.membrane.shape
        scaled_size = (scale * i_size, scale * j_size)
        self.image = ti.Vector.field(3, float, shape=scaled_size)
        return scaled_size

    def show(self, abs_highest: float = 1.0):
        """
        Show current state of the membrane.

        :param abs_highest: The highest absolute value found in the dataset.
        Default = 1.0
        :type abs_highest: float
        """
        gui = ti.GUI("Membrane", res=self.image.shape, fast_gui=True)  # type: ignore
        self.draw(abs_highest)
        while gui.running:
            gui.set_image(self.image)
            gui.show()

    def animate(self, frequency: float, abs_highest: float = 1.0, dt: float = 1 / 60):
        """
        Animate and show a vibrating eigenvector.

        :param frequency: The frequency of the eigenmode. Affects the
        speed that the membrane changes.
        :type frequency: float
        :param abs_highest: The highest absolute value found in the dataset.
        Default = 1.0
        :type abs_highest: float
        :param dt: The time step size. Default = 1/60, which at 60 fps could
        be considered real time.
        :type dt: float
        """
        gui = ti.GUI("Membrane animated", res=self.image.shape, fast_gui=True)  # type: ignore
        t = 0.0
        while gui.running:
            self.animated_draw(abs_highest, frequency, t)
            gui.set_image(self.image)
            gui.show()
            t += dt

    def save_animation(
        self,
        frequency: float,
        frames: int,
        mode_index: int,
        abs_highest: float = 1.0,
        dt: float = 1 / 60,
        output_dir: str = "./local",
    ):
        """
        Animate a vibrating eigenvector and save it to a file.

        :param frequency: The frequency of the eigenmode. Affects the
        speed that the membrane changes.
        :type frequency: float
        :param frames: The length of the animation in frames.
        :type frames: int
        :param mode_index: The number of the eigenmode animated, used in the
        name of the saved video.
        :type mode_index: int
        :param abs_highest: The highest absolute value found in the dataset.
        Default = 1.0
        :type abs_highest: float
        :param dt: The time step size. Default = 1/60, which at 60 fps could
        be considered real time.
        :type dt: float
        :param output_dir: The folder to save the video in. Default: "./local"
        :type output_dir: str
        """
        # gui = ti.GUI("Membrane animated", res=self.image.shape, fast_gui=True, show_gui=False)  # type: ignore
        video_manager = ti.tools.VideoManager(
            output_dir=output_dir,
            framerate=60,
            automatic_build=True,
            video_filename=f"{self.name}_{self.membrane.shape[0]}_{self.membrane.shape[1]}_{mode_index}",
        )
        t = 0.0

        for _ in range(frames):
            self.animated_draw(abs_highest, frequency, t)
            # gui.set_image(self.image)
            video_manager.write_frame(self.image)
            t += dt
        video_manager.make_video(mp4=False, gif=True)

        frame_folder = os.path.join(output_dir, "frames")
        for fn in os.listdir(frame_folder):
            if fn.endswith(".png"):
                os.remove(frame_folder + "/" + fn)
        os.rmdir(frame_folder)

    def __str__(self) -> str:
        return self.name


@ti.data_oriented
class TaichiSolver:
    """
    Class to find the eigenfrequencies of supplied 2D membranes using matrix
    solving methods.
    """

    def __init__(self, membrane: Membrane) -> None:
        self.membrane = membrane
        self.adjecency_matrix = ti.field(
            ti.f64, shape=(membrane.cell_count, membrane.cell_count)
        )
        self.spatial_constant = self.membrane.h**2
        self.ranked_membrane = membrane.cell_number
        self.construct_adjecency_matrix()

    @ti.kernel
    def construct_adjecency_matrix(self):
        """Create the matrix of which the eigenvectors will be found."""
        for i, j in self.ranked_membrane:
            if self.ranked_membrane[i, j] == 0:
                continue
            cell_rank = self.ranked_membrane[i, j] - 1
            self.adjecency_matrix[cell_rank, cell_rank] = -4.0 / self.spatial_constant
            for ni, nj in ti.static([[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]):
                if not (
                    ni < 0
                    or ni >= self.ranked_membrane.shape[0]
                    or nj < 0
                    or nj >= self.ranked_membrane.shape[1]
                    or self.ranked_membrane[ni, nj] == 0
                ):
                    neighbor_rank = self.ranked_membrane[ni, nj] - 1
                    self.adjecency_matrix[cell_rank, neighbor_rank] = (
                        1.0 / self.spatial_constant
                    )

    def solve(self):
        """
        Use the `scipy.linalg.eigh()` function to solve the eigenvalue problem.
        Results are stored in `self.eigenvalues` and `self.eigenvectors`.
        """
        self.eigenvalues, self.eigenvectors = eigh(self.adjecency_matrix.to_numpy())
        self.eigenvalues = np.flip(self.eigenvalues)
        self.eigenvectors = np.rot90(self.eigenvectors)

    def show_eigenvector(self, vector_index: int):
        """
        Display the structure of the given eigenmode.

        :param vector_index: The index of the eigenmode to show.
        :type vector_index: int
        """
        eigenfrequency, absolute_highest_value = self.prep_display(vector_index)
        self.membrane.show(abs_highest=absolute_highest_value)

    def animate_eigenvector(self, vector_index: int, dt: float = 1 / 60):
        """
        Display the structure of the given eigenmode, and animate it according to
        its eigenfrequency.

        :param vector_index: The index of the eigenmode to show.
        :type vector_index: int
        :param dt: The time step size. Default = 1/60, which at 60 fps could
        be considered real time.
        :type dt: float
        """
        eigenfrequency, absolute_highest_value = self.prep_display(vector_index)
        self.membrane.animate(eigenfrequency, abs_highest=absolute_highest_value, dt=dt)

    def save_eigenvector_animation(
        self,
        vector_index: int,
        frames: int,
        dt: float = 1 / 60,
        output_dir: str = "local",
    ):
        """
        Animate the structure of the given eigenmode according to
        its eigenfrequency and save it to a file.

        :param vector_index: The index of the eigenmode to show.
        :type vector_index: int
        :param dt: The time step size. Default = 1/60, which at 60 fps could
        be considered real time.
        :type dt: float
        :param output_dir: The folder to save the video in.
        :type output_dir: str
        """
        eigenfrequency, absolute_highest_value = self.prep_display(vector_index)
        self.membrane.save_animation(
            eigenfrequency,
            frames,
            vector_index,
            abs_highest=absolute_highest_value,
            dt=dt,
            output_dir=output_dir,
        )

    def prep_display(self, vector_index: int) -> tuple[float, float]:
        """
        A helper function to prepare the `self.membrane` to show the given
        eigenvector.

        :param vector_index: The index of the eigenmode to show.
        :type vector_index:
        :return: A tuple containing: the eigenfrequency of the vector, and
        the absolute highest value found in the eigenvector.
        :rtype: tuple[float, float]
        """
        vector_to_show = self.eigenvectors[vector_index]
        eigenfrequency = np.sqrt(-self.eigenvalues[vector_index])

        self.load_into_membrane(vector_to_show)
        absolute_highest_value = max(abs(min(vector_to_show)), abs(max(vector_to_show)))
        return eigenfrequency, absolute_highest_value

    def load_into_membrane(self, eigenvector: NDArray):
        """
        Copies the given eigenvector into the `self.membrane`.

        :param eigenvector: The eigenvector to copy into the membrane.
        :type eigenvector: NDArray
        """
        for i, j in np.ndindex(self.membrane.membrane.shape):
            if self.ranked_membrane[i, j] == 0:
                continue
            self.membrane.membrane[i, j] = eigenvector[self.ranked_membrane[i, j] - 1]


@ti.data_oriented
class SparseSolver(TaichiSolver):
    """
    Class to find the eigenfrequencies of supplied 2D membranes using sparse
    matrix solving methods.
    """

    def solve(self, k: int = 15):
        """
        Use the `scipy.linalg.eigh()` function to solve the eigenvalue problem.
        Results are stored in `self.eigenvalues` and `self.eigenvectors`.

        :param k: The amount of eigenvectors generated.
        :type k: float

        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh
        - https://stackoverflow.com/questions/11083660/python-eigenvectors-differences-among-numpy-linalg-scipy-linalg-and-scipy-spar?rq=3
        - https://en.wikipedia.org/wiki/Lanczos_algorithm
        """
        self.eigenvalues, self.eigenvectors = eigsh(
            self.adjecency_matrix.to_numpy(), k=k, which="SM"
        )
        self.eigenvalues = np.flip(self.eigenvalues)
        self.eigenvectors = np.rot90(self.eigenvectors)


if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    mem = Membrane.circle(1.0, 0.02, ui_scale=10)
    print(f"{mem.cell_count = }")
    spsolver = SparseSolver(mem)
    spsolver.solve()
    spsolver.save_eigenvector_animation(6, 60 * 30, dt=10.0)
