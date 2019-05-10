import keras
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import conv2d

def main():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    Y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    Y_test  = keras.utils.np_utils.to_categorical(y_test, 10)

    X_train = X_train.reshape(
        X_train.shape[0], 28, 28, 1).astype(np.float32) / 255.0
    X_test = X_test.reshape(
        X_test.shape[0], 28, 28, 1).astype(np.float32) / 255.0

    beta = 1
    input_shape = X_train[0].shape
    sigma = 1.0
    step_size_init = 0.001
    batch_size = 128
    max_epochs = 100
    dropout = True

    linear = None
    encoder_input = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y_ = tf.placeholder(tf.float32, shape=(None, 10))

    x = tf.layers.Conv2D(4, (3,3), 
            activation=tf.nn.relu,
            padding="same")(encoder_input)
    x = tf.layers.Conv2D(4, (3,3), 
            activation=tf.nn.relu,
            padding="same")(x)
    x = tf.layers.MaxPooling2D((2,2), strides=(2,2))(x)
    encoder_output = information_dropout(x, dropout=dropout)
    x = tf.layers.Flatten()(encoder_output)
    x = tf.layers.Dense(256, activation=tf.nn.relu)(x)
    decoder_output = tf.layers.Dense(10, activation=None)(x)

    cross_entropy_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=decoder_output))
    
    if dropout:
        kl_penalties = [tf.reduce_sum(kl) / float(batch_size) 
                        for kl in tf.get_collection("kl_terms")]
        kl_cost = tf.add_n(kl_penalties)
        global_cost = cross_entropy_cost + beta*kl_cost
    else:
        global_cost = cross_entropy_cost

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
            print("Epoch: {} => {}/{}".format(
                e, np.mean(train_accuracy), np.mean(test_accuracy)))

        # get the encoder output
        test_iterator = batch_iterator(X_test, Y_test)
        codewords = []
        true_labels = []
        for X, Y in test_iterator:
            codewords.append(
                sess.run(encoder_output, feed_dict={encoder_input: X}))
            true_labels.append(np.argmax(Y, axis=1))
        print np.array(codewords).shape
        dropout_stub = "with" if dropout else "no"
        np.save("codewords_{}_dropout".format(dropout_stub), np.array(codewords))
        np.save("true_labels_{}_dropout".format(dropout_stub), np.array(true_labels))

        test_iterator = batch_iterator(X_test, Y_test)
        codewords = []
        for X, Y in test_iterator:
            codewords.append(
                sess.run(decoder_output, feed_dict={encoder_input: X}))
        print np.array(codewords).shape
        dropout_stub = "with" if dropout else "no"
        np.save("logits_{}_dropout".format(dropout_stub), np.array(codewords))


def batch_iterator(X, Y, batch_size=128):
    for ix in range(0, X.shape[0], batch_size):
        next_X = X[ix:ix+batch_size,:]
        next_Y = Y[ix:ix+batch_size,:]
        yield (next_X, next_Y)


def sample_lognormal(mean, sigma=None, sigma0=1.):
    e = tf.random_normal(tf.shape(mean), mean = 0., stddev = 1.)
    return tf.exp(mean + sigma * sigma0 * e)


def information_dropout(inputs, stride=2, max_alpha=0.7, sigma0=1.0, dropout=True):
    num_outputs = inputs.get_shape()[-1]

    # compute the noiseless output using a convolutional layer
    network = conv2d(inputs,
        num_outputs=num_outputs,
        kernel_size=3,
        activation_fn=tf.nn.relu,
        stride=stride)

    if dropout:
        with tf.variable_scope(None, "information_dropout"):
            alpha = max_alpha * conv2d(inputs,
                num_outputs=num_outputs,
                kernel_size=3,
                stride=stride,
                activation_fn=tf.sigmoid,
                scope="alpha")
            alpha = 1e-3 + max_alpha*alpha

            kl = -tf.log(alpha/(max_alpha + 1e-3))
            tf.add_to_collection("kl_terms", kl)
        
        e = sample_lognormal(mean=tf.zeros_like(network), sigma=alpha, sigma0=sigma0)
    else:
        e = 1.0

    return network * e


if __name__=="__main__":
    main()
