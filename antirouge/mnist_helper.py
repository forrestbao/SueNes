from keras.datasets import mnist
import tensorflow as tf

def __test_load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train
    len(x_train[y_train == 1])
    len(x_train)
    x_train.shape

    image_1 = x_train[y_train == 1]
    tf.nn.nce_loss

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = (original_dim, )
    latent_dim = 2

    vae, encoder, decoder = causal_vae_model(input_shape, 2)
    # vae.summary()
    # encoder.summary()
    history = vae.fit(x_train, epochs=15, batch_size=128,
                      shuffle=True, validation_data=(x_test, None),
                      callbacks=[EarlyStopping(monitor='val_loss', min_delta=0,
                                               patience=3, verbose=0, mode='auto')])
    # plot_history(history, 'causal/history.png')
    
    data = (x_test, y_test)
    plot_results((encoder, decoder),
                 data,
                 batch_size=128,
                 model_name="causal")


def __test():
    sampled_values = tf.nn.log_uniform_candidate_sampler(
        true_classes=math_ops.cast(train_labels, dtypes.int64),
        num_true=1,
        num_sampled=64,
        unique=True,
        range_max=50000)
    sampled_values
