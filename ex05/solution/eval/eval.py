import matplotlib as mpl
from matplotlib import rc as rc
from matplotlib import pyplot as plt

rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.style.use('bmh')

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

problem_size = [
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
print(df)

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
