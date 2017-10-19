import tensorflow as tf
import numpy as np
def xavier_init(fan_in, fan_out, constant=1):
    #xavier initialization of network weights
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high= constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float64)