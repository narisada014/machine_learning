#  http://www.sist.ac.jp/~kanakubo/research/statistic/juukaiki.html
#  上記の行列版の重回帰分析を参考にしなければ本の数式が導出できない
#  http://www.snap-tck.com/room04/c01/stat/stat07/stat0701.html
#  このスライドも参考になる
#  http://nlp.dse.ibaraki.ac.jp/~shinnou/zemi2012/kernel/vector-bibun.pdf

import numpy as np
from scipy import linalg


class LinearRegression:
    def __init__(self):
        self.w_ = None

    # 訓練データによる学習, Xは入力訓練データ, tは出力訓練データ
    def fit(self, X, t):
        # 100行2列の行列がXの場合は以下のような出力になる
        # >> > X.shape
        # (100, 2)

        # np.ones(X.shape[0])は100行一列のベクトルを作る。そこにXをc_で結合してやることで1列目が全て1の行列Xが完成する。
        Xtil = np.c_[np.ones(X.shape[0]), X]  # 要素1からなるベクトルを加えた行列X
        A = np.dot(Xtil.T, Xtil)  # # 重回帰分析での求める行列式の係数を除いた部分Xの転置行列×Xを表している
        b = np.dot(Xtil.T, t)  # 重回帰分析での求める行列式の係数を除いた部分Xの転置行列×yベクトル(出力)を表している。
        # Aw = bをとく行列式になるのでlinalg.solveでとく事ができる
        # linalg.solveはAX=BのXを求める事ができるので上記のような重回帰分析に落とし込める式を解く事ができる。
        self.w_ = linalg.solve(A, b)

    # 入力値Xが与えられた時の予測される出力を求める
    # 上の訓練でw_が決定しているのでXwを解く単純な問題となる。
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)

