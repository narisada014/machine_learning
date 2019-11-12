import numpy as np
import matplotlib.pyplot as plt
import warnings
import polyreg
import linearreg
#  raise LinAlgError('Matrix is singular.')
#  この例外の対応もいつかする（逆行列が求められない場合発生する）


def f(x):
    return 1 / (1 + x)


def sample(n):
    x = np.random.random(n) * 5
    y = f(x)
    return x, y


# 0.01刻みごとの配列を0~5までで作成
xx = np.arange(0, 5, 0.01)
np.random.seed(0)
# この方法で0~5の0.01刻み＝500要素の全てが0の配列を用意してやる
y_poly_sum = np.zeros(len(xx))
# array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0.])
y_lin_sum = np.zeros(len(xx))
n = 1000
# 警告の制御（今回は無視）
warnings.filterwarnings("ignore")
# 1000回繰り返す
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
    # 線形回帰予測
    y_lin = lin.predict(xx.reshape(-1, 1))
    y_lin_sum += y_lin

print(y_poly_sum)
# 真の値
plt.plot(xx, f(xx), label="truth",
         color="k", linestyle="solid")
# 多項式回帰の予測値(n: 回数 で平均するので真の値と数値のレンジが揃う)
plt.plot(xx, y_poly_sum / n, label="polynomial reg",
         color="k", linestyle="dotted")
# 線形回帰の予測値
plt.plot(xx, y_lin_sum / n, label="linear reg.",
         color="k", linestyle="dashed")
plt.legend()
plt.show()
