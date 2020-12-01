import matplotlib as mpl
from matplotlib import rc as rc
from matplotlib import pyplot as plt

rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.style.use('bmh')

import pandas as pd

data = [[100 , 1196938     , 1.67093],
[200  , 10429803    , 1.53407],
[300  , 24751539    , 2.18168],
[400  , 46317528    , 2.76353],
[500  , 90040211    , 2.77654],
[600  , 156633734   , 2.75803],
[700  , 248539772   , 2.76012],
[800  , 375961078   , 2.72369],
[900  , 530355986   , 2.7491],
[1000 , 730676262   , 2.73719],
[1100 , 982741209   , 2.70875],
[1200 , 1468651903  , 2.35318],
[1300 , 2107968175  , 2.08447],
[1400 , 4005978469  , 1.36995],
[1500 , 4401971320  , 1.5334],
[1600 , 5970232772  , 1.37214],
[1700 , 7083584663  , 1.38715],
[1800 , 8499171649  , 1.37237],
[1900 , 10635954837 , 1.28978],
[2000 , 10551356235 , 1.51639],
[2100 , 15129504667 , 1.22423],
[2200 , 16504030714 , 1.29035],
[2300 , 21867738449 , 1.11278],
[2400 , 26758385848 , 1.03325],
[2500 , 33152235784 , 0.942621]]

df = pd.DataFrame(data=data, columns=["dim", "t", "GFLOPS"])
df["size"] = df["dim"] * df["dim"] * 8
print(df)

plt.figure()
plt.plot(df["size"], df["t"], linestyle='--', marker='^')
plt.xlabel("Problem Size [Byte]")
plt.ylabel("Execution Time [ns]")
plt.title("Execution Time Matrix Multiply")
plt.xscale('log')
plt.yscale('log')
plt.savefig("mmul_time.pdf")

plt.figure()
plt.plot(df["size"], df["GFLOPS"], linestyle='--', marker='^')
plt.vlines(196608, 0, 3, linestyles='--', colors='r', label="L1 Cache: 192KiB")
plt.vlines(3.146e6, 0, 3, linestyles='--', colors='g', label="L2 Cache: 3MiB")
plt.vlines(3.355e7, 0, 3, linestyles='--', colors='y', label="L3 Cache: 32MiB")
plt.xlabel("Problem Size [Byte]")
plt.ylabel("Operations [GFLOP/s]")
plt.title("Throughput Matrix Multiply")
plt.xscale('log')
plt.legend()
plt.savefig("mmul_through.pdf")
