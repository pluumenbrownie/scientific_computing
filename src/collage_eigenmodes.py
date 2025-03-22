from eigenmodes import Membrane, SparseSolver, TaichiSolver
import taichi as ti
import numpy as np
import math as mt
import matplotlib.pyplot as plt


def eigenmode_collage(path: str):
    """Plots a few eigenmodes for the three shapes of membrane in a grid."""
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
    plt.close()


def eigenfrequency_plot(path: str):
    """
    Plot the relation between the size of the membrane and
    the found eigenfrequencies.
    """
    h = 0.02
    mode_index = 1

    Ls = np.linspace(0.5, 1.5, 11)
    square_freqs = []
    rectangle_freqs = []
    circle_freqs = []

    for L in Ls:
        square = Membrane.square(L, h)
        rectangle = Membrane.rectangle(L, 2 * L, h)
        circle = Membrane.circle(L, h)
        for shape, shape_freqs in zip(
            [square, rectangle, circle], [square_freqs, rectangle_freqs, circle_freqs]
        ):
            solver = SparseSolver(shape)
            solver.solve(k=mode_index + 1)
            shape_freqs.append(mt.sqrt(-solver.eigenvalues[mode_index]))
            print(f"Plotted {shape} at {L = }")

    plt.figure(figsize=(5, 4), layout="compressed")
    plt.plot(Ls, square_freqs, "o", label="Square")
    plt.plot(Ls, rectangle_freqs, "o", label="Rectangle")
    plt.plot(Ls, circle_freqs, "o", label="Circle")
    plt.title("Eigenfrequency scaling")
    plt.xlabel("L")
    plt.ylabel("frequency")
    plt.xticks(np.arange(0.5, 1.6, 0.2))
    plt.legend()
    plt.savefig(path)
    plt.close()


def animate_eigenvectors(folder: str):
    """Animates and save a few eigenmodes."""
    L = 1.0
    h = 0.02
    frames = 60 * 30
    save_modes = list(range(0, 7))
    animation_speed = 10.0

    shape = Membrane.circle(L, h, ui_scale=4)
    solver = SparseSolver(shape)
    solver.solve(k=7)

    for mode in save_modes:
        solver.save_eigenvector_animation(
            mode, frames, dt=animation_speed, output_dir=folder
        )


if __name__ == "__main__":
    ti.init()
    # eigenmode_collage("local/eigenmode_collage.pdf")
    # eigenfrequency_plot("local/eigenfrequencies.pdf")
    animate_eigenvectors("local")
