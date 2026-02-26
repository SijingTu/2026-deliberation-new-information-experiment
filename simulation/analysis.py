# read the stats.txt file and plot the mean, variance, and std_dev against lambda

import matplotlib.pyplot as plt
import numpy as np

with open("stats.txt", "r") as f:
    lines = f.readlines()
    lambdas = []
    means = []
    variances = []
    std_devs = []
    for line in lines:
        lam, mean, variance, std_dev = line.split()
        lambdas.append(float(lam))
        means.append(float(mean))
        variances.append(float(variance))
        std_devs.append(float(std_dev))

# plt.plot(lambdas, means, label="mean")
plt.plot(lambdas, variances, label="variance")
# plt.plot(lambdas, std_devs, label="std_dev")
plt.legend()
plt.savefig("stats.png")