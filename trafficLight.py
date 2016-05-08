# coding: utf-8

# ロジスティック回帰について以下を参考に作成
# 参考：http://yaju3d.hatenablog.jp/entry/2016/04/21/022519
# 必要なモジュールを読み込む
import numpy as np
import tensorflow as tf

# TensorFlow でロジスティック回帰する

# 1. 学習したいモデルを記述する
# 入力変数と出力変数のプレースホルダを生成
# 信号機の入力パラメータを[青,黄,赤]（1ならその色がON）とする
x = tf.placeholder(tf.float32, shape=(None, 3), name="x")
# 教師データは1なら進む、0は止まれ
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# モデルパラメータ
a = tf.Variable(-10 * tf.ones((3, 1)), name="a")
b = tf.Variable(200., name="b")
# モデル式
u = tf.matmul(x, a) + b
y = tf.sigmoid(u)

# 2. 学習やテストに必要な関数を定義する
# 誤差関数(loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(u, y_))
# 最適化手段(最急降下法)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# 3. 実際に学習処理を実行する
# (1) 訓練データを生成する
# 複数の色がONの信号機はおかしいので止まれとして扱う
train_x = np.array([[1., 0., 0.], [0., 1., 1.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.], [1., 1., 1.], [1., 0., 1.]])
train_y = np.array([1., 0., 0., 0., 0., 0., 0.]).reshape(7, 1)
print("x=", train_x)
print("y=", train_y)

# (2) セッションを準備し，変数を初期化
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# (3) 最急勾配法でパラメータ更新 (1000回更新する) 
for i in range(3000): 
    _, l, a_, b_ = sess.run([train_step, loss, a, b], feed_dict={x: train_x, y_: train_y})
    if (i + 1) % 100 == 0:
        print("step=%3d, a1=%6.2f, a2=%6.2f, b=%6.2f, loss=%.2f" % (i + 1, a_[0], a_[1], b_, l))

# (4) 学習結果を出力
est_a, est_b = sess.run([a, b], feed_dict={x: train_x, y_: train_y})
print("Estimated: a1=%6.2f, a2=%6.2f, b=%6.2f" % (est_a[0], est_a[1], est_b))

# 4. 新しいデータに対して予測する
# (1) 新しいデータを用意
# 青、黄、赤がそれぞれONのデータを用意する
new_x = np.array([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]).reshape(3, 3)

# (2) 学習結果をつかって，予測実施
new_y = sess.run(y, feed_dict={x: new_x})
print(new_y)

# 5. 後片付け
# セッションを閉じる
sess.close()


