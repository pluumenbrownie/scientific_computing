import taichi as ti
from collage_eigenmodes import eigenmode_collage


ti.init()

eigenmode_collage("figures/eigenmode_collage.pdf")
