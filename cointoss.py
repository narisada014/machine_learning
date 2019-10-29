import numpy as np
import matplotlib.pyplot as plt


def cointoss(n, m):
    a = []
    for _ in range(m):
        r = np.random.randint(2, size=n)
        a.append(r.sum())
    return a


np.random.seed(0)
fig, axes = plt.subplots(1, 2)
a = cointoss(100, 100)
axes[0].hist(a, range=(30, 70), bins=50, color="k")
a = cointoss(10000, 1000)
axes[1].hist(a, range=(4800, 5200), bins=50, color="k")
plt.show()
