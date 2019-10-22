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


xmin, xmax, ymin, ymax = -3, 3, -3, 3

algos = []
initial = np.array([1, 1])
alphas = [0.1, 0.2]
for alpha in alphas:
    algo = gradient.GradientDescent(f, df, alpha)
    algo.solve(np.array(initial))
    algos.append(algo)

# 等高線の描画
xs = np.linspace(xmin, xmax, 300)
ys = np.linspace(ymin, ymax, 300)
xmesh, ymesh = np.meshgrid(xs, ys)
xx = np.r_[xmesh.reshape(1, -1), ymesh.reshape(1, -1)]

# 何個の図にするかを定義（1行2列の図にする）
fig, ax = plt.subplots(1, 2)
# axは以下のような配列 => 二つのプロット
# [<matplotlib.axes._subplots.AxesSubplot object at 0x10ed71e80>
#  <matplotlib.axes._subplots.AxesSubplot object at 0x1140f7e48>]


levels = [-3, -2.9, -2.8, -2.6, -2.4, -2.2, -2, -1, 0, 1, 2, 3, 4]

# 表の設定を行い描画する。
for i in range(2):
    ax[i].set_xlim((xmin, xmax))
    ax[i].set_ylim((ymin, ymax))
    ax[i].set_title("alpha={}".format(alphas[i]))
    ax[i].scatter(initial[0], initial[1], color="k", marker="o")
    ax[i].plot(algos[i].path_[:, 0], algos[i].path_[:, -1], color="k", linewidth=1.5)
    ax[i].contour(xs, ys, f(xx).reshape(xmesh.shape), levels=levels,
                  colors="k", linestyles="dotted")


plt.show()

# 結果としてalphaが初期値の10倍異常なので線の収束までの移動が暴れる
