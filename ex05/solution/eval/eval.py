import matplotlib as mpl
from matplotlib import rc as rc
from matplotlib import pyplot as plt

rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.style.use('bmh')

import numpy as np
import pandas as pd

thread_count = [
[1 , 2763.42],
[2  , 449.154 ],
[4  , 74.5626 ],
[8  , 22.0544 ],
[16 , 15.6459 ],
[32 , 15.3111 ]
]

df = pd.DataFrame(data=thread_count, columns=["threads", "time"])

plt.figure()
plt.plot(df["threads"], df["time"], linestyle='--', marker='^')
plt.xlabel("Number of threads per Block")
plt.ylabel("Execution Time [ms]")
plt.title("Optimizing the Thread Count (1024x1024 Matrix)")
plt.yscale('log')
plt.savefig("thread_count.pdf")

sh_problem_size = [
[4*(2**4)**2  , 0.02745  , 0.02031  , 0.03325 ]  ,
[4*(2**5)**2  , 0.030121 , 0.0204   , 0.03463 ]  ,
[4*(2**6)**2  , 0.034281 , 0.02216  , 0.03626 ]  ,
[4*(2**7)**2  , 0.04096  , 0.02979  , 0.04312 ]  ,
[4*(2**8)**2  , 0.05472  , 0.019438 , 0.091791 ] ,
[4*(2**9)**2  , 0.303603 , 0.800567 , 0.246152 ] ,
[4*(2**10)**2 , 2.01958  , 1.14962  , 0.716656 ] ,
[4*(2**11)**2 , 15.3268  , 3.70113  , 3.53799 ]  ,
[4*(2**12)**2 , 118.52   , 12.4173  , 11.6666 ]  ,
[4*(2**13)**2 , 747.777  , 48.3087  , 44.3199 ]  ,
[4*(2**14)**2 , 5497.34  , 186.355  , 175.345 ]  ,
]

df = pd.DataFrame(data=sh_problem_size, columns=["size", "tmult", "D2H", "H2D"])

df['total'] = df['tmult'] + df['D2H'] + df['H2D']

plt.figure()
plt.plot(df["size"], df["tmult"], linestyle='--', marker='^', label='mmul')
plt.plot(df["size"], df["D2H"], linestyle='--', marker='^', label='D2H')
plt.plot(df["size"], df["H2D"], linestyle='--', marker='^', label='H2D')
plt.plot(df["size"], df["total"], linestyle='--', marker='^', label='total')
plt.xlabel("Problem Size [Byte]")
plt.ylabel("Execution Time [ms]")
plt.title("Optimizing the Problem Size with Shared Memory")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig("sh_problem_size.pdf")

sh_thread_count = [
[1 , 22369.6 ],
[2  , 7186.19 ],
[4  , 5026.29 ],
[8  , 3090.49 ],
[16 , 1747.64 ],
[32 , 691.025 ]
]

df = pd.DataFrame(data=sh_thread_count, columns=["threads", "time"])

plt.figure()
plt.plot(df["threads"], df["time"], linestyle='--', marker='^')
plt.xlabel("Number of threads per Block")
plt.ylabel("Execution Time [ms]")
plt.title("Optimizing the Thread Count with Shared Memory (8192x8192 Matrix)")
plt.yscale('log')
plt.savefig("sh_thread_count.pdf")

problem_size = [
[4*(2**4)**2  , 0.02228  , 0.01993  , 0.029871 ] ,
[4*(2**5)**2  , 0.0272   , 0.02009  , 0.036271 ] ,
[4*(2**6)**2  , 0.02944  , 0.02242  , 0.03778 ]  ,
[4*(2**7)**2  , 0.029201 , 0.03071  , 0.0398 ]   ,
[4*(2**8)**2  , 0.06988  , 0.207802 , 0.277583 ] ,
[4*(2**9)**2  , 0.375333 , 0.775428 , 0.277583 ] ,
[4*(2**10)**2 , 2.42303  , 1.43569  , 0.856578 ] ,
[4*(2**11)**2 , 19.8834  , 3.75653  , 3.07188 ]  ,
[4*(2**12)**2 , 153.092  , 12.562   , 11.8904 ]  ,
[4*(2**13)**2 , 1617.42  , 47.2959  , 45.1102 ]  ,
[4*(2**14)**2 , 20997.2  , 187.128  , 175.372 ]  ,
]

df = pd.DataFrame(data=problem_size, columns=["size", "tmult", "D2H", "H2D"])

df['total'] = df['tmult'] + df['D2H'] + df['H2D']

plt.figure()
plt.plot(df["size"], df["tmult"], linestyle='--', marker='^', label='mmul')
plt.plot(df["size"], df["D2H"], linestyle='--', marker='^', label='D2H')
plt.plot(df["size"], df["H2D"], linestyle='--', marker='^', label='H2D')
plt.plot(df["size"], df["total"], linestyle='--', marker='^', label='total')
plt.xlabel("Problem Size [Byte]")
plt.ylabel("Execution Time [ms]")
plt.title("Optimizing the Problem Size")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig("problem_size.pdf")

shared = pd.DataFrame(data=sh_problem_size, columns=["size", "tmult", "D2H", "H2D"])
shared['total'] = shared['tmult'] + shared['D2H'] + shared['H2D']
unopt = pd.DataFrame(data=problem_size, columns=["size", "tmult", "D2H", "H2D"])
unopt['total'] = unopt['tmult'] + unopt['D2H'] + unopt['H2D']

plt.figure()
plt.plot(shared["size"], shared["tmult"], linestyle='--', marker='^', label='with Shared Mem')
plt.plot(unopt["size"], unopt["tmult"], linestyle='--', marker='^', label='w/o Shared Mem')
plt.xlabel("Problem Size [Byte]")
plt.ylabel("Execution Time [ms]")
plt.title("Comparing Optimized to Unoptimized Execution Time")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig("compare.pdf")

plt.figure()
plt.plot(shared["size"], np.abs(shared["tmult"] - unopt["tmult"]), linestyle='--', marker='^', label='with Shared Mem')
plt.xlabel("Problem Size [Byte]")
plt.ylabel(r"$t_{\textrm{shmem}} - t_{\textrm{unopt}}$ [ms]")
plt.title("Difference in Execution Time Optimized vs Unoptimized")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig("diff.pdf")

time_cpu = 730676262 * 10**-6
time_gpu_unopt = unopt["tmult"][6]
time_gpu_opt = shared["tmult"][6]
total_gpu_unopt = unopt["total"][6]
total_gpu_opt = shared["total"][6]
print("speedup unoptimized (no data movement):\t\t {}".format(time_cpu / time_gpu_unopt))
print("speedup optimized (no data movement):\t\t {}".format(time_cpu / time_gpu_opt))
print("speedup unoptimized (with data movement):\t {}".format(time_cpu / total_gpu_unopt))
print("speedup optimized (with data movement):\t\t {}".format(time_cpu / total_gpu_opt))
