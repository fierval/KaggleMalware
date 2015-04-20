import numpy as np
import matplotlib.pylab as plt

from numba import cuda, uint8, int32, uint32, jit
from timeit import default_timer as timer

@cuda.jit('void(uint8[:], int32, int32[:], int32[:])')
def lbp_kernel(input, neighborhood, powers, h):
    i = cuda.grid(1)
    r = 0
    if i < input.shape[0] - 2 * neighborhood:
        i += neighborhood
        for j in range(i - neighborhood, i):
            if input[j] >= input[i]:
                r += powers[j - i + neighborhood]
    
        for j in range(i + 1, i + neighborhood + 1):
            if input[j] >= input[i]:
                r += powers[j - i + neighborhood - 1]

        cuda.atomic.add(h, r, 1)

def extract_1dlbp_gpu(input, neighborhood, d_powers):
    maxThread = 512

    blockDim = maxThread
    d_input = cuda.to_device(input)

    hist = np.zeros(2 ** (2 * neighborhood), dtype='int32')
    gridDim = (len(input) - 2 * neighborhood + blockDim) / blockDim

    d_hist = cuda.to_device(hist)

    lbp_kernel[gridDim, blockDim](d_input, neighborhood, d_powers, d_hist)
    d_hist.to_host()
    return hist

def extract_1dlbp_gpu_debug(input, neighborhood, powers, res):
    maxThread = 512
    blockDim = maxThread
    gridDim = (len(input) - 2 * neighborhood + blockDim) / blockDim
    
    for block in range(0, gridDim):
        for thread in range(0, blockDim):
            r = 0
            i = blockDim * block + thread
            if i < input.shape[0] - 2 * neighborhood:
                i += neighborhood
                for j in range(i - neighborhood, i):
                    if input[j] >= input[i]:
                        r += powers[j - i + neighborhood]
    
                for j in range(i + 1, i + neighborhood + 1):
                    if input[j] >= input[i]:
                        r += powers[j - i + neighborhood - 1]

                res[r] += 1
    return res

@jit("int32[:](uint8[:], int64, int32[:], int32[:])", nopython=True)
def extract_1dlbp_cpu_jit(input, neighborhood, powers, res):
    maxThread = 512
    blockDim = maxThread
    gridDim = (len(input) - 2 * neighborhood + blockDim) / blockDim
    
    for block in range(0, gridDim):
        for thread in range(0, blockDim):
            r = 0
            i = blockDim * block + thread
            if i < input.shape[0] - 2 * neighborhood:
                i += neighborhood
                for j in range(i - neighborhood, i):
                    if input[j] >= input[i]:
                        r += powers[j - i + neighborhood]
    
                for j in range(i + 1, i + neighborhood + 1):
                    if input[j] >= input[i]:
                        r += powers[j - i + neighborhood - 1]

                res[r] += 1
    return res

def extract_1dlbp_cpu(input, neighborhood, p):
    """
    Extract the 1d lbp pattern on CPU
    """
    res = np.zeros(1 << (2 * neighborhood))
    for i in range(neighborhood, len(input) - neighborhood):
        left = input[i - neighborhood : i]
        right = input[i + 1 : i + neighborhood + 1]
        both = np.r_[left, right]
        res[np.sum(p [both >= input[i]])] += 1
    return res

X = np.arange(3, 7)
X = 10 ** X
neighborhood = 4

cpu_times = np.zeros(X.shape[0])
cpu_times_simple = cpu_times.copy()
cpu_times_jit = cpu_times.copy()
gpu_times = np.zeros(X.shape[0])

p = 1 << np.array(range(0, 2 * neighborhood), dtype='int32')
d_powers = cuda.to_device(p)

for i, x in enumerate(X):
    input = np.random.randint(0, 256, size = x).astype(np.uint8)

    print "Length: {0}".format(x)
    print "--------------"

    start = timer()
    h_cpu = extract_1dlbp_cpu(input, neighborhood, p)
    cpu_times[i] = timer() - start
    print "Finished on CPU: time: {0:3.5f}s".format(cpu_times[i])

    res = np.zeros(1 << (2 * neighborhood), dtype='int32')
    start = timer()
    h_cpu_simple = extract_1dlbp_gpu_debug(input, neighborhood, p, res)
    cpu_times_simple[i] = timer() - start
    print "Finished on CPU (simple): time: {0:3.5f}s".format(cpu_times_simple[i])

    res = np.zeros(1 << (2 * neighborhood), dtype='int32')
    start = timer()
    h_cpu_jit = extract_1dlbp_cpu_jit(input, neighborhood, p, res)
    cpu_times_jit[i] = timer() - start
    print "Finished on CPU (numba: jit): time: {0:3.5f}s".format(cpu_times_jit[i])

    start = timer()
    h_gpu = extract_1dlbp_gpu(input, neighborhood, d_powers)
    gpu_times[i] = timer() - start
    print "Finished on GPU: time: {0:3.5f}s".format(gpu_times[i])
    print "All h_cpu == h_gpu: ", (h_cpu_jit == h_gpu).all() and (h_cpu_simple == h_cpu_jit).all() and (h_cpu == h_cpu_jit).all()
    print ''

f = plt.figure(figsize=(10, 5))

plt.plot(X, cpu_times, label = "CPU")
plt.plot(X, cpu_times_simple, label = "CPU non-vectorized")
plt.plot(X, cpu_times_jit, label = "CPU jit")
plt.plot(X, gpu_times, label = "GPU")
plt.yscale('log')
plt.xscale('log')
plt.xlabel('input length')
plt.ylabel('time, sec')
plt.legend()
plt.show()
