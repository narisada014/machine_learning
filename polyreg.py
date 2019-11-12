import linearreg
import numpy as np


# 多項式回帰
class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree

    def fit(self, x, y):
        x_pow = []
        # reshapeは第二引数の列数に配列を変換するreshape(5, 2)ならば10列のarrayは5行2列になる
        xx = x.reshape(len(x), 1)
        # xxは以下のような形
        # [[5.48813504]
        #  [7.15189366]
        #  [6.02763376]
        #  [5.44883183]
        #  [4.23654799]
        #  [6.45894113]
        #  [4.37587211]
        #  [8.91773001]
        #  [9.63662761]
        #  [3.83441519]]
        for i in range(1, self.degree + 1):
            # ここで多項式回帰の元になるn乗の計算を実現している
            x_pow.append(xx**i)
        # ベクトルを横につなぐ
        mat = np.concatenate(x_pow, axis=1)
        linreg = linearreg.LinearRegression()
        linreg.fit(mat, y)
        self.w_ = linreg.w_

    def predict(self, x):
        r = 0
        for i in range(self.degree + 1):
            r += x**i * self.w_[i]
        return r