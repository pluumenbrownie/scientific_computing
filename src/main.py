from gray_scott_experiments import pyplot_combined
from numpy.random import seed


# Fix numpy seed to prevent unnessecary figure updates
seed(12650662)


# Gray-Scott collage
pyplot_combined(save_location="./figures/gray_scott_collage.pdf")
