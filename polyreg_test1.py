import polyreg
import linearreg
import numpy as np
import matplotlib.pyplot as plt


# データ生成
np.random.seed(0)


def f(x):
    return 1 + 2 * x


x = np.random.random(10) * 10
# >>> x
# array([5.48813504, 7.15189366, 6.02763376, 5.44883183, 4.23654799,
#        6.45894113, 4.37587211, 8.91773001, 9.63662761, 3.83441519])

y = f(x) + np.random.randn(10)
# >>> y = (1 + 2 * x) + np.random.randn(10)
# >>> y
# array([13.47034915, 15.09862906, 13.36833522, 11.04356792,  6.92010617,
#        14.57150086, 10.61618042, 18.093295  , 22.54300983,  7.2144647 ])

# 多項式回帰, degreeが10で与えられる
model = polyreg.PolynomialRegression(10)
model.fit(x, y)

plt.scatter(x, y, color="k")
plt.ylim([y.min() - 1, y.max() + 1])
xx = np.linspace(x.min(), x.max(), 300)
yy = np.array([model.predict(u) for u in xx])
plt.plot(xx, yy, color="k")

# 線形回帰
model = linearreg.LinearRegression()
model.fit(x, y)
b, a = model.w_
x1 = x.min() - 1
x2 = x.max() + 1
plt.plot([x1, x2], [f(x1), f(x2)], color="k", linestyle="dashed")

plt.show()