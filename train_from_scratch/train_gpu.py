import cv2
import numpy as np
from numba import jit,cuda
import math 

@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1

ary = np.zeros((64,64))
d_ary = cuda.to_device(ary)
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(ary.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(ary.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
increment_a_2D_array[blockspergrid, threadsperblock](d_ary)