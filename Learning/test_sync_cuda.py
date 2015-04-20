import numpy as np
from numba import cuda, int32

maxThread = 512

@cuda.jit('void(int32[:])')
def idx_kernel(arr):
    s = cuda.shared.array(shape=maxThread, dtype=int32)

    idx = cuda.grid(1)
    if idx < arr.shape[0]:
        s[cuda.threadIdx.x] = 1

    cuda.syncthreads()

    if idx < arr.shape[0]:
        cuda.atomic.add(arr, s[cuda.threadIdx.x], 1)

def launch_idx_test(arr):
    blockDim = maxThread
    gridDim = (arr.shape[0] + blockDim) / blockDim

    d_arr = cuda.to_device(arr)

    idx_kernel[gridDim, blockDim](d_arr)

    d_arr.to_host()
    return (arr == np.arange(arr.shape[0])).all()

arr = np.zeros(10000000).astype('int32')