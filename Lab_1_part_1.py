import time

from numba import cuda
import numpy as np # Arrays in Python
from matplotlib import pyplot as plt # Plotting library
from math import sin, cos, pi

# Repeatable results
np.random.seed(0)

# Define the observed signal
signal = np.random.normal(size=512, loc=0, scale=1).astype(np.float32)

@cuda.jit
def dft_parallel(samples, aantalsamples, frequencies):
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    sample = samples[x]


    for k in range(frequencies.shape[1]):
        prod_real = sample * (cos(2 * pi * k * x / aantalsamples))
        prod_im   = sample * (-1) * sin(2 * pi * k * x / aantalsamples)
        cuda.atomic.add(frequencies[0], k, prod_real)
        cuda.atomic.add(frequencies[1], k, prod_im)



# Define the sampling rate and observation time
SAMPLING_RATE_HZ = 100
TIME_S = 5 # Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S

# Define sample times
x = np.linspace(0, TIME_S, int(N), endpoint=False)

sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05

frequencies = np.zeros([2, int(N/2+1)])
#print("\nMatrix c : \n", frequencies)

dft_parallel[1, 1](sig_sum, N, frequencies)

sumfreqs = np.zeros(int(N/2+1), dtype=np.complex)
for k in range(len(frequencies[0])):
    sumfreqs[k] = frequencies[0][k] + frequencies[1][k] * 1j

#print("\nMatrix c : \n", sumfreqs)

# Plot to evaluate whether the results are as expected
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

# Calculate the appropriate X-axis for the frequency components
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

# Plot all of the signal components and their sum
for sig in sigs:
    ax1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
ax1.plot( x, sig_sum )
ax1.set_title("Sum Signal Components")
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Time(s)")

# Plot the frequency components
ax2.plot( xf, abs(sumfreqs), color='C3' )
ax2.set_title("Frequency Components")
ax2.set_ylabel("DFT")
ax2.set_xlabel("Frequency components")
fig.suptitle('GPU Graphs', fontsize=16)
plt.savefig('gpugraphs')

plt.show()

def synchronous_kernel_timeit(kernel, number=1, repeat=1):
    times = []
    for r in range(repeat):
        start = time.time()
        for n in range(number):
            kernel()
            cuda.synchronize()  # Do not queue up, instead wait for all previous kernel launches to finish executing
        stop = time.time()
        times.append((stop - start) / number)
    return times[0] if len(times) == 1 else times

t_par = synchronous_kernel_timeit(lambda: dft_parallel[5,1000](sig_sum, N, frequencies), number=10)
print(t_par)