import tensorflow as tf
import tensorflow_hub as hub

## for tensorflow 2.0.0

if tf.__version__ == '2.0.0':
    hub.Module = hub.load
    tf.logging = tf.compat.v1.logging
    tf.ConfigProto = tf.compat.v1.ConfigProto
    tf.Session = tf.compat.v1.Session
    tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
    tf.tables_initializer = tf.compat.v1.tables_initializer
    tf.train.RMSPropOptimizer = tf.keras.optimizers.RMSprop