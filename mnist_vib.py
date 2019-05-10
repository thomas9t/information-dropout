import sys
import keras
import numpy as np
import tensorflow as tf

vib = bool(int(sys.argv[1]))

def batch_iterator(X, Y, batch_size=128):
    for ix in range(0, X.shape[0], batch_size):
        next_X = X[ix:ix+batch_size,:]
        next_Y = Y[ix:ix+batch_size,:]
        yield (next_X, next_Y)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
Y_train = keras.utils.np_utils.to_categorical(y_train, 10)
Y_test  = keras.utils.np_utils.to_categorical(y_test, 10)

beta = 1e-5
input_shape = (None, 784)
sigma = 1.0
encoder_output_size = 3
step_size_init = 0.001
batch_size = 128
max_epochs = 100

linear = None
encoder_input = tf.placeholder(tf.float32, shape=input_shape)
y_ = tf.placeholder(tf.float32, shape=(None, 10))

x = tf.layers.Dense(784, activation=tf.nn.relu)(encoder_input)
x = tf.layers.Dense(1024, activation=tf.nn.relu)(encoder_input)
x = tf.layers.Dense(1024, activation=tf.nn.relu)(encoder_input)
x = tf.layers.Dense(2*encoder_output_size, activation=linear)(x)

if vib:
    print "===========> Using VIB <==========="
    mu = x[:,:encoder_output_size]
    sigma = tf.nn.softplus(x[:,encoder_output_size:]) + 1
    encoder_output = mu + tf.random_normal(tf.shape(mu), stddev=1)*sigma
else:
    print "===========> No VIB <==========="
    encoder_output = x

x = tf.layers.Dense(256, activation=tf.nn.relu)(encoder_output)
decoder_output = tf.layers.Dense(10, activation=linear)(x)

if vib:
    logdet_a = tf.reduce_sum(tf.log(sigma), axis=1)
    logdet_b = tf.cast(0, tf.float32)
    inv_sigma_a = tf.divide(1, sigma)
    inv_sigma_b = tf.divide(1, 1)
    a = tf.reduce_sum(inv_sigma_a, axis=1)
    b = tf.reduce_sum(tf.multiply(tf.multiply(mu, inv_sigma_a), mu), axis=1)
    c = logdet_a - logdet_b
    kl_div = 0.5*(a + b - encoder_output_size + c)
else:
    kl_div = 0

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=decoder_output)
)

global_cost = cross_entropy + beta*kl_div

global_step = tf.Variable(0, trainable=False)
step_size = tf.train.exponential_decay(
    step_size_init, global_step, 1000, 0.9, staircase=True)
train_step = tf.train.AdamOptimizer(
    step_size).minimize(global_cost, global_step=global_step)
correct_prediction = tf.cast(
    tf.equal(tf.argmax(decoder_output, 1), tf.argmax(y_, 1)), tf.float32)
accuracy_op = tf.reduce_mean(correct_prediction)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    for e in range(10):
        train_iterator = batch_iterator(X_train, Y_train)
        for X, Y in train_iterator:
            sess.run(train_step, feed_dict={encoder_input: X, y_: Y})
        
        # calculate accuracy
        train_iterator = batch_iterator(X_train, Y_train)
        train_accuracy = []
        for X, Y in train_iterator:
            train_accuracy.append(
                sess.run(accuracy_op, feed_dict={encoder_input: X, y_: Y}))

        test_iterator = batch_iterator(X_test, Y_test)
        test_accuracy = []
        for X, Y in test_iterator:
            test_accuracy.append(
                sess.run(accuracy_op, feed_dict={encoder_input: X, y_: Y}))
        print "Epoch: {} => {}/{}".format(
            e, np.mean(train_accuracy), np.mean(test_accuracy))

    # get the encoder output
    test_iterator = batch_iterator(X_test, Y_test)
    codewords = []
    true_labels = []
    for X, Y in test_iterator:
        codewords.append(
            sess.run(encoder_output, feed_dict={encoder_input: X}))
        true_labels.append(np.argmax(Y, axis=1))
    print np.array(codewords).shape
    vib_stub = "with" if vib else "no"
    np.save("temp/codewords{}_{}_vib".format(encoder_output_size, vib_stub), np.array(codewords))
    np.save("temp/true_labels_{}_vib".format(vib_stub), np.array(true_labels))
