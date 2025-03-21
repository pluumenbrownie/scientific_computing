import taichi as ti
from collage_eigenmodes import *


ti.init()

eigenmode_collage("figures/eigenmode_collage.pdf")
eigenfrequency_plot("figures/eigenfrequencies.pdf")
