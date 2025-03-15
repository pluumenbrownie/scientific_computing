import taichi as ti
import numpy as np
from numpy.typing import NDArray
from typing import Any, Self


@ti.data_oriented
class Membrane:
    """
    A model for a vibrating membrane or drum.
    """

    membrane: ti.ScalarField | ti.MatrixField
    ignore: ti.ScalarField | ti.MatrixField

    def __init__(self, membrane: Any) -> None:
        self.membrane = membrane
        self.ignore = ti.field(shape=self.membrane.shape, dtype=int)

    @classmethod
    def square(cls, L: int) -> Self:
        """
        Returns a square membrane.
        """
        membrane = cls(ti.field(shape=(L, L), dtype=float))
        return membrane

    @classmethod
    def rectangle(cls, L1: int, L2: int) -> Self:
        """
        Returns a square membrane.
        """
        membrane = cls(ti.field(shape=(L1, L2), dtype=float))
        return membrane

    @classmethod
    def circle(cls, L: int) -> Self:
        """
        Returns a square membrane.
        """
        membrane = cls(ti.field(shape=(L, L), dtype=float))
        membrane.fill_circle()
        return membrane

    # @ti.kernel
    # def fill_circle(self):

    #     for i, j in self.ignore:
    #         if (i**2 + j**2 > )

    @ti.kernel
    def draw(self, scale: int):
        for i, j in self.image:
            if self.ignore[i // scale, j // scale] == 1:
                self.image[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])
            else:
                self.image[i, j] = (
                    self.membrane[i // scale, j // scale] / 2.0 + 0.5
                ) * ti.Vector([1.0, 1.0, 1.0, 1.0])
                self.image[i, j][3] = 1.0

    def show(self, scale: int = 1, speed: int = 1):
        """
        Run diffusion steps and show the resulting diffusions live until closed.

        # Inputs:
        - scale: How much the shown video should be scaled. Default = 1
        - speed: How many diffusion steps should be taken between frame updates. Default = 1
        """
        i_size, j_size = self.membrane.shape
        scaled_size = (scale * i_size, scale * j_size)
        gui = ti.GUI("Membrane example", res=scaled_size, fast_gui=True)
        self.image = ti.Vector.field(4, float, shape=scaled_size)

        while gui.running:
            self.draw(scale)
            gui.set_image(self.image)
            gui.show()


if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    mem = Membrane.circle(100)
    print(mem.membrane)
    print(mem.membrane.shape)
    mem.show(scale=4)
