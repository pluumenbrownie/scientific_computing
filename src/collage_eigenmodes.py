from eigenmodes import Membrane, SparseSolver, TaichiSolver
import taichi as ti
import numpy as np
import math as mt
import matplotlib.pyplot as plt


def eigenmode_collage(path: str):
    L = 1.0
    h = 0.02
    mode_indexes = list(range(0, 7))
    shapes = [
        Membrane.square(L, h),
        Membrane.rectangle(2 * L, L, h),
        Membrane.circle(L, h),
    ]

    fig = plt.figure(figsize=(8, 4), layout="compressed")
    axes = fig.subplots(
        nrows=len(shapes),
        ncols=len(mode_indexes),
        subplot_kw={"xticks": [], "yticks": []},
    )
    cmap = plt.get_cmap("viridis")
    # the "bad" data will be the cells outside of circle
    cmap.set_bad(color="white")
    fig.suptitle("Membrane eigenmodes with eigenfrequencies", fontsize=18)

    for row, shape in enumerate(shapes):
        solver = SparseSolver(shape)
        solver.solve(k=max(mode_indexes) + 1)
        for col, mode in enumerate(mode_indexes):
            eigenvector = solver.eigenvectors[mode]
            solver.load_into_membrane(eigenvector)
            # mark the cells outside of the boundary
            plotted_data = np.ma.masked_where(
                shape.cell_number.to_numpy() == 0, shape.membrane.to_numpy()
            )
            # plotted data is rotated 90 to align with the taichi display
            axes[row, col].imshow(np.rot90(plotted_data), cmap=cmap)
            axes[row, col].set_xlabel(f"{mt.sqrt(-solver.eigenvalues[mode]):.5f}")
            axes[row, col].set_frame_on(False)
            if col == 0:
                if row == 0:
                    axes[row, col].set_ylabel(f"Square")
                elif row == 1:
                    axes[row, col].set_ylabel(f"Rectangle")
                if row == 2:
                    axes[row, col].set_ylabel(f"Circle")
        print(f"Plotted {shape}")
    plt.savefig(path)


if __name__ == "__main__":
    ti.init()
    eigenmode_collage("local/eigenmode_collage.pdf")
