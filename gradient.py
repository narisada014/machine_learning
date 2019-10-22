import numpy as np


# f: 最小化したい関数
# df: fの導関数
# n変数の最適化を行う場合、引数はnの1次元の配列
class GradientDescent:
    def __init__(self, f, df, alpha=0.01, eps=1e-6):
        self.f = f
        self.df = df
        self.alpha = alpha
        self.eps = eps
        self.path = None

    def solve(self, init):
        x = init
        path = []
        grad = self.df(x)
        path.append(x)
        #  epsがマイナスの場合も含めた一般化
        while (grad ** 2).sum() > self.eps ** 2:
            x = x - self.alpha * grad
            grad = self.df(x)
            path.append(x)
        self.path_ = np.array(path)
        self.x_ = x
        self.opt_ = self.f(x)