import linearreg
import numpy as np
import matplotlib.pyplot as plt


n = 100
scale = 10
np.random.seed(0)
X = np.random.random((n, 2)) * scale
w0 = 1
w1 = 2
w2 = 3
y = w0 + w1 * X[:, 0] + w2 * X[:, 1] + np.random.randn(n)

model = linearreg.LinearRegression()
model.fit(X, y)
print("係数:", model.w_)
# predictの引数はベクトルか行列である
# 上記のy式でXはw1とw2にかかっており、このyで学習した、
# モデルに行列を渡すなら、2列あれば良いと考えられるので以下のように2次元ベクトルを引数として計算してやる。
print("(1, 1)に対する予測値:", model.predict(np.array([1, 1])))

# ここ以降は可視化のコードになる。
xmesh, ymesh = np.meshgrid(np.linspace(0, scale, 20),
                           np.linspace(0, scale, 20))
zmesh = (model.w_[0] + model.w_[1] * xmesh.ravel() + model.w_[2] * ymesh.ravel()).reshape(xmesh.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color="k")
ax.plot_wireframe(xmesh, ymesh, zmesh, color="r")
plt.show()
