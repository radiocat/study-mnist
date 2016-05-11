#coding: UTF-8
# あらかじめgithubからinput_data.pyや関連ファイルを取得しておく
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist
import input_data
# MNISTデータをダウンロードして読み込む
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# TensorFlowのインポート
import tensorflow as tf

# MNISTデータを入れる変数
# 28x28のMNIST画像を1pxごとに1行784列のベクトルに入れる
x = tf.placeholder(tf.float32, [None, 784])
# 重み
# MNISTデータの各pxごとに0-9の重みをつける（初期値0をセットしておく）
W = tf.Variable(tf.zeros([784, 10]))
# バイアス
# （初期値は0）
b = tf.Variable(tf.zeros([10]))
# ソフトマックス回帰の実行式
y = tf.nn.softmax(tf.matmul(x, W) + b)


# 教師データ
y_ = tf.placeholder(tf.float32, [None, 10])
# 損失関数の計算（交差エントロピー）
# reduction_indices…各ユニットで計算させる
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# GradientDescentOptimizer…勾配硬化法、学習率は0.5
# minimizez…損失関数である交差エントロピーを最小化させる
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 学習の実行

# 重み、バイアスを初期化する処理を実行
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 1000回学習
for i in range(1000):
    # ミニバッチサイズは100
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# モデルの評価
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



