import matplotlib as mpl
from matplotlib import rc as rc
from matplotlib import pyplot as plt

rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.style.use('bmh')

import numpy as np
import pandas as pd

thread_count = [
[2**9  , 90.235872 , 79.654782],
[2**10 , 119.204424 , 106.707097],
[2**11 , 234.636794 , 212.079334],
[2**12 , 464.895227 , 423.679851],
[2**13 , 938.359035 , 855.973025],
[2**14 , 1994.05815 , 1891.2021],
[2**15 , 5749.2063 , 5585.97923],
[2**16 , 20938.4004 , 20662.7413],
[2**17 , 78288.2266 , 77039.1151],
[2**18 , 295932.086 , 291133.747]
]

df = pd.DataFrame(data=thread_count, columns=["elems", "time_unopt", "time_opt"])
print(df)

plt.figure()
plt.plot(df["elems"], df["time_unopt"], linestyle='--', marker='^', label="Unoptimized")
# plt.plot(df["elems"], df["time_opt"], linestyle='--', marker='x', label="Optimized")
plt.xlabel("Number of Elements")
plt.ylabel("Execution Time [ms]")
plt.title("Execution Time for unoptimized n-Body Kernel")
plt.yscale('log')
plt.xscale('log')
# plt.legend()
plt.savefig("../plot/time_vs_elems_unopt.pdf")

plt.figure()
# plt.plot(df["elems"], df["time_unopt"], linestyle='--', marker='^', label="Unoptimized")
plt.plot(df["elems"], df["time_opt"], linestyle='--', marker='^', label="Optimized")
plt.xlabel("Number of Elements")
plt.ylabel("Execution Time [ms]")
plt.title("Execution Time for optimized n-Body Kernel")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig("../plot/time_vs_elems_opt.pdf")

plt.figure()
plt.plot(df["elems"], df["time_unopt"], linestyle='--', marker='^', label="Unoptimized")
plt.plot(df["elems"], df["time_opt"], linestyle='--', marker='x', label="Optimized")
plt.xlabel("Number of Elements")
plt.ylabel("Execution Time [ms]")
plt.title("Execution Time for unoptimized and optimized n-Body Kernel")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig("../plot/time_vs_elems_both.pdf")
