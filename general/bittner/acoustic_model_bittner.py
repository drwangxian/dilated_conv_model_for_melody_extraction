import tensorflow as tf
from self_defined import get_name_scope


def bn_relu_fn(inputs):
    assert get_name_scope() != ''

    outputs = inputs

    outputs = tf.keras.layers.BatchNormalization(
        name=get_name_scope() + 'bn',
        scale=False
    )(outputs)

    outputs = tf.keras.layers.ReLU(
        name=get_name_scope() + 'relu'
    )(outputs)

    return outputs


def create_acoustic_model_fn():

    inputs = tf.keras.Input([None, 360, 6], batch_size=1, name='hcqt')
    outputs = inputs

    c_layers = [[128, 5], [64, 5], [64, 3], [64, 3]]
    for layer_idx, (n_features, k_size) in enumerate(c_layers):
        with tf.name_scope('layer_{}'.format(layer_idx)):

            outputs = tf.keras.layers.Conv2D(
                name=get_name_scope() + 'conv',
                kernel_size=k_size,
                use_bias=False,
                padding='SAME',
                filters=n_features
            )(outputs)

            outputs = bn_relu_fn(outputs)

    with tf.name_scope('layer_4'):
        outputs = tf.keras.layers.Conv2D(
            name=get_name_scope() + 'conv',
            kernel_size=[3, 70],
            use_bias=False,
            padding='SAME',
            filters=8
        )(outputs)

        outputs = bn_relu_fn(outputs)

    with tf.name_scope('output'):
        outputs = tf.keras.layers.Dense(
            name=get_name_scope() + 'dense',
            units=1,
            use_bias=True
        )(outputs)
        outputs = tf.squeeze(outputs, [0, -1])
        outputs.set_shape([None, 360])

    model = tf.keras.Model(inputs, outputs, name='bittner')

    return model


if __name__ == '__main__':

    model = create_acoustic_model_fn()
    model.summary(line_length=150)

    for w in model.trainable_variables:
        print(w.name, w.shape)

    inputs = tf.random.normal([1, 1200, 360, 6])
    outputs = model(inputs)
    print(outputs.shape)

