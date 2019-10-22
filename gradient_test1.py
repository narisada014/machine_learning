import numpy as np
import matplotlib.pyplot as plt
import gradient


# 関数を定義
def f(xx):
    x = xx[0]
    y = xx[1]
    return 5 * x**2 - 6 * x * y + 3 * y**2 + 6 * x - 6 * y


# 導関数を定義(左からxで微分したもの、yで微分したもの)
def df(xx):
    x = xx[0]
    y = xx[1]
    return np.array([10 * x - 6 * y + 6, -6 * x + 6 * y - 6])


algo = gradient.GradientDescent(f, df)
initial = np.array([1, 1])
algo.solve(initial)
print(algo.x_)
print(algo.opt_)

plt.scatter(initial[0], initial[1], color="k", marker="o")

# algo.path_ は以下のような状態
# [[1.00000000e+00 1.00000000e+00]
#  [9.00000000e-01 1.06000000e+00]
#  [8.13600000e-01 1.11040000e+00]
#  ...
#  [3.57605424e-07 1.00000050e+00]
#  [3.51613943e-07 1.00000049e+00]
#  [3.45722846e-07 1.00000048e+00]]

# 収束の過程を描画
plt.plot(algo.path_[:, 0], algo.path_[:, 1], color="k", linewidth=1.5)

# 等高線の描画
xs = np.linspace(-2, 2, 300)
ys = np.linspace(-2, 2, 300)
xmesh, ymesh = np.meshgrid(xs, ys)
xx = np.r_[xmesh.reshape(1, -1), ymesh.reshape(1, -1)]
levels = [-3, -2.9, -2.8, -2.6, -2.4, -2.2, -2, -1, 0, 1, 2, 3, 4]

plt.contour(xs, ys, f(xx).reshape(xmesh.shape),
            levels=levels,
            colors="k", linestyles="dotted")

plt.show()
