import numpy as np


# 符号関数 (sign function) (配列または行列の要素の絶対値)-yの値が0以上かどうかで1, 0, -1を返す
# yはL1ノルムの偏微分を行うのでラッソ回帰の場合ラムダになる
def soft_thresholding(x, y):
    return np.sign(x) * max(abs(x) - y, 0)


class Lasso:
    def __init__(self, lambda_, tol=0.0001, max_iter=1000):
        self.lambda_ = lambda_  # ノルムの係数ラムダ(この値をいじってモデルの最適化を行う。)
        self.tol = tol  # 収束許容度を示す。この値より小さくなれば計算を終える
        self.max_iter = max_iter  # 無限ループしないように。
        self.w_ = None

    def fit(self, X, t):
        print("行列X: {}".format(X))
        print("行列Xの形: {}".format(X.shape))
        n, d = X.shape
        # >> > X = np.array([1, 1, 1])
        # >> > X
        # array([1, 1, 1])
        # >> > X.shape
        # (3, 1)
        # nは入力値の行列の行方向の要素数を表す
        # dは入力値の行列の列方向の要素数を表す

        # dは行列Xの行方向の次元数。 1を足しているのはw0のため
        self.w_ = np.zeros(d + 1)
        # avgl1はL1ノルムの平均（avg）という意味。
        # 初期値として0.(float型の0をいれておく)
        # forの繰り返しの中でw_が更新され、記録されるたびに更新されていく
        avgl1 = 0.
        for _ in range(self.max_iter):
            avgl1_prev = avgl1
            self._update(n, d, X, t)  # wの値の更新
            # w_ が更新されるのでavgl1が数値で表される
            avgl1 = np.abs(self.w_).sum() / self.w_.shape[0]
            # 今回の更新によるL1ノルムの平均と前回のL1ノルムの平均の差を計算して、
            # トレランス(収束許容度)より小さくなっているかを判定して、小さくなっていれば学習を止める
            # 更新がほとんど系に効いていないという事なので「もう十分」ということ
            if abs(avgl1 - avgl1_prev) <= self.tol:
                break

    # 各x要素に掛かるwの値の更新を行う。(t: 出力値 X: 入力のデータ行列, d: 入力のデータの次元)
    def _update(self, n, d, X, t):
        # w0の計算を行う
        self.w_[0] = (t - np.dot(X, self.w_[1:])).sum() / n
        w0vec = np.ones(n) * self.w_[0]
        # kは0から始まる
        for k in range(d):
            ww = self.w_[1:]
            # w_はもともとXの次元+1次元されているのでw0が計算された後の計算では、
            # w0の次元を削除してwwとしてやればXと行列計算可能な次元になる。
            ww[k] = 0  # j not equal kなのでwwのk番目の要素は0になるようにしてやる
            q = np.dot(t - w0vec - np.dot(X, ww), X[:, k])
            # >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            # >>> a
            # array([[1, 2, 3],
            #        [4, 5, 6],
            #        [7, 8, 9]])
            # >>> a[:, 1] # a[:, k]はk列目の数値でそれ以外をスライスした配列を作成する
            # array([2, 5, 8])

            # 式変形後の分母の行列計算
            r = np.dot(X[:, k], X[:, k])
            # 符号関数によるwの値の評価(kは0スタート。初回はw1を更新する)
            self.w_[k + 1] = soft_thresholding(q / r, self.lambda_)

    # モデルによる予測を行う。
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)
