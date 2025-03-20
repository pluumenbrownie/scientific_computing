from eigenmodes import Membrane, SparseSolver
import matplotlib.pyplot as plt


def eigenmode_collage():
    L = 1.0
    h = 0.05
    mode_indexes = list(range(0, 7))
    shapes = [
        Membrane.square(L, h),
        Membrane.rectangle(2 * L, L, h),
        Membrane.circle(L, h),
    ]

    fig = plt.figure(figsize=(4, 10), layout="compressed")
    axes = fig.subplots(
        nrows=len(shapes),
        ncols=len(mode_indexes),
        subplot_kw={"xticks": [], "yticks": []},
    )

    for row, shape in enumerate(shapes):
        solver = SparseSolver(shape)
        for col, mode in enumerate(mode_indexes):
            pass
