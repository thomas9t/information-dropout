import tensorflow as tf
from tensorflow.contrib.layers import conv2d


def sample_lognormal(mean, sigma=None, sigma0=1.):
    """
    Samples from a log-normal distribution using the reparametrization
    trick so that we can backprogpagate the gradients through the sampling.
    By setting sigma0=0 we make the operation deterministic (useful at testing time)
    """
    e = tf.random_normal(tf.shape(mean), mean = 0., stddev = 1.)
    return tf.exp(mean + sigma * sigma0 * e)


def information_dropout(inputs, stride = 2, max_alpha = 0.7, sigma0 = 1.):
    """
    An example layer that performs convolutional pooling
    and information dropout at the same time.
    """
    num_ouputs = inputs.get_shape()[-1]
    # Creates a convolutional layer to compute the noiseless output
    network = conv2d(inputs,
        num_outputs=num_outputs,
        kernel_size=3,
        activation_fn=tf.nn.relu,
        stride=stride)
    # Computes the noise parameter alpha for the new layer based on the input
    with tf.variable_scope(None,'information_dropout'):
        alpha = max_alpha * conv2d(inputs,
            num_outputs=num_outputs,
            kernel_size=3,
            stride=stride,
            activation_fn=tf.sigmoid,
            scope='alpha')
        # Rescale alpha in the allowed range and add a small value for numerical stability
        alpha = 0.001 + max_alpha * alpha
        # Similarly to variational dropout we renormalize so that
        # the KL term is zero for alpha == max_alpha
        kl = - tf.log(alpha/(max_alpha + 0.001))
        tf.add_to_collection('kl_terms', kl)
    e = sample_lognormal(mean=tf.zeros_like(network), sigma=alpha, sigma0=sigma0)
    # Noisy output of Information Dropout
    return network * e

### BUILD THE NETWORK
# ...
# Computes the KL divergence term in the cost function
kl_terms = [ tf.reduce_sum(kl)/batch_size for kl in tf.get_collection('kl_terms') ]
# Normalizes by the number of training samples to make
# the parameter beta comparable to the beta in variational dropout
Lz = tf.add_n(kl_terms)/N_train
# Lx is the cross entropy loss of the network
Lx = cross_entropy_loss
# The final cost
cost = Lx + beta * Lz