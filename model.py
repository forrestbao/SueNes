#!/usr/bin/env python3


def build_model():
    """The model contains:
    """
    model = Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64,
                           # input_shape=(768,),
                           activation='relu'))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    # model.add(layers.Dense(10, activation='softmax'))
    # output the score
    model.add(layers.Dense(1, activation='sigmoid'))

    # x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = GlobalMaxPooling1D()(x)
    # x = Dense(128, activation='relu')(x)
    
    return model

def build_glove_model(embedding_layer):
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_ARTICLE_LENGTH + MAX_SUMMARY_LENGTH,),
                           dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1)(x)

    model = Model(sequence_input, preds)
    return model

def build_model_test():
    # apply a layer multiple times
    i = Input(shape=(13,100), dtype='float32')
    i.shape
    l = Dense(512, input_dim=(100))
    tf.concat([l(t) for t in tf.split(i, 13, 1)], 1)
    x = tf.reshape(i, shape=(100,13))
    #
    # original:
    # 100 -> 512
    i = Dense(512, input_dim=(100))(i)

    
def build_use_model(use_embed):
    """Since USE can only used CPU, I'd better not getting it into the
Model layers.
    """
    sequence_input = Input(shape=(13,), dtype=tf.string)
    # (?, 13)
    sequence_input
    
    # Simply doing this is not working:
    # >>> embedded_sequences = use_embed(sequence_input)
    # So instead, split the tensors and concatenate afterwards
    in_sub_tensors = tf.split(sequence_input, 13, 1)
    in_sub_tensors
    # takes time to construct
    # (?) to (512)
    out_sub_tensors = [use_embed(tf.reshape(t, [-1]))
                       for t in in_sub_tensors]
    embedded_sequences = tf.concat([tf.reshape(t, (-1, 1, 512))
                                 for t in out_sub_tensors], axis=1)
    # (?, 13, 512)
    embedded_sequences

    # testing use_embed:
    # >>> holder = tf.placeholder(tf.string, shape=(None))
    # >>> holder.shape
    # >>> holder
    # >>> similarity_message_encodings = use_embed(holder)
    # (13, 512)

    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1)(x)

    model = Model(sequence_input, preds)
    return model
