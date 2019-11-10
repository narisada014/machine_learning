import numpy as np
from scipy import linalg


class RidgeRegression:
    def __init__(self, lambda_=1.):
        self.lambda_ = lambda_
        self.w_ = None

    # 入力に対してwを計算して求める。
    def fit(self, X, t):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        c = np.eye(Xtil.shape[1])
        # >>> np.eye(3) 単位行列の生成
        # array([[1., 0., 0.],
        #        [0., 1., 0.],
        #        [0., 0., 1.]])
        A = np.dot(Xtil.T, Xtil) + self.lambda_ * c
        b = np.dot(Xtil.T, t)
        # linalg.solveはAX=BのXを求める事ができるので上記のような重回帰分析に落とし込める式を解く事ができる。
        self.w_ = linalg.solve(A, b)

    def predict(self, X):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)