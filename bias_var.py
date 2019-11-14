import numpy as np
import matplotlib.pyplot as plt
import warnings
import polyreg
import linearreg
# バイアス(予測と真の値のずれ)とバリアンス(予測値の分散)については以下に導出が掲載されている


def f(x):
    return 1 / (1 + x)


def sample(n):
    x = np.random.random(n) * 5
    y = f(x)
    return x, y


xx = np.arange(0, 5, 0.01)
np.random.seed(0)
y_poly_sum = np.zeros(len(xx))
y_poly_sum_sq = np.zeros(len(xx))
y_lin_sum = np.zeros(len(xx))
y_lin_sum_sq = np.zeros(len(xx))
y_true = f(xx)
n = 1000
warnings.filterwarnings("ignore")
for _ in range(n):
    # 1. 適当な5点をとる
    x, y = sample(5)
    # 多項式回帰学習
    poly = polyreg.PolynomialRegression(4) # 多項式の次数(5点だから4次関数)
    poly.fit(x, y)
    # 線形回帰学習
    lin = linearreg.LinearRegression()
    lin.fit(x, y)
    # 多項式回帰予測(0~5の全ての点で)
    y_poly = poly.predict(xx)
    # 多項式回帰予測：結果を用意した配列に格納してやる
    y_poly_sum += y_poly
    # 真の値と予測値の差の2乗を配列に格納
    y_poly_sum_sq += (y_poly - y_true)**2
    # 線形回帰予測
    y_lin = lin.predict(xx.reshape(-1, 1))
    y_lin_sum += y_lin
    # 真の値と予測値の差の2乗を配列に格納
    y_lin_sum_sq += (y_lin - y_true)**2

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_title("Linear reg.")
ax2.set_title("Polynomial reg.")
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax1.fill_between(xx, 0, (y_lin_sum / n - y_true)**2,
                 color="0.2", label="bias")
ax1.fill_between(xx, (y_lin_sum / n - y_true)**2, y_lin_sum_sq / n, color="0.7", label="variance")
ax1.legend(loc="upper left")
ax2.fill_between(xx, 0, (y_poly_sum / n - y_true)**2,
                 color="0.2", label="bias")
ax2.fill_between(xx, (y_poly_sum / n - y_true)**2, y_poly_sum_sq / n, color="0.7", label="variance")
ax2.legend(loc="upper left")
plt.show()
