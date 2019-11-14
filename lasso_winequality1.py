import lasso
import numpy as np
import csv


# データ読み込み
Xy = []
with open("winequality-red.csv") as fp:
    for row in csv.reader(fp, delimiter=";"):
        Xy.append(row)  # 1行ごとデータを配列に格納してやる
Xy = np.array(Xy[1:], dtype=np.float64)


# 訓練用データとテスト用データに分割する
np.random.seed(0)
np.random.shuffle(Xy)
train_X = Xy[:-1000, :-1]  # 出力データ以外の入力値の次元のみを切り出している。
train_y = Xy[:-1000, -1]  # 出力データの次元を切り出している
test_X = Xy[-1000:, :-1]  # 出力データ以外の入力値の次元のみを切り出している。
test_y = Xy[-1000:, -1]  # 出力データの次元を切り出している

# ハイパーパラメータを変えながら学習させて結果表示
for lambda_ in [1., 0.1, 0.01]:
    model = lasso.Lasso(lambda_)
    model.fit(train_X, train_y)
    y = model.predict(test_X)
    print("--- lambda = {} ---".format(lambda_))
    print("coefficients:")
    print(model.w_)
    mse = ((y - test_y)**2).mean()  # この値は小さいほど良いモデルと言える（誤差が小さいということ）
    print("MSE: {: .3f}".format(mse))

#  ラッソ回帰の方が線形回帰と比べて良い結果になっている

