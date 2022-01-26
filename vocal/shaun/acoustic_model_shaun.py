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


def create_acoustic_model_fn(reg=1e-4):

    inputs = tf.keras.Input([None, 500], batch_size=1, name='cqt-shaun')
    outputs = inputs
    outputs = outputs[..., None]

    with tf.name_scope('local'):
        for layer_idx in range(4):
            with tf.name_scope('layer_{}'.format(layer_idx)):

                outputs = tf.keras.layers.Conv2D(
                    name=get_name_scope() + 'conv',
                    kernel_size=[5, 5] if layer_idx == 0 else [3, 5],
                    dilation_rate=[2 ** layer_idx, 1],
                    padding='SAME',
                    use_bias=False,
                    activation=None,
                    filters=16
                )(outputs)
                outputs.set_shape([None, None, 500, None])

                outputs = bn_relu_fn(outputs)

                if layer_idx > 0:
                    outputs = tf.keras.layers.Dropout(
                        name=get_name_scope() + 'dropout',
                        rate=.2
                    )(outputs)

    with tf.name_scope('global'):

        outputs = tf.pad(outputs, [[0, 0], [0, 0], [240, 60], [0, 0]])
        outputs = tf.keras.layers.Conv2D(
            name=get_name_scope() + 'conv',
            kernel_size=[1, 97],
            dilation_rate=[1, 5],
            padding='VALID',
            use_bias=False,
            activation=None,
            filters=128,
            kernel_regularizer=tf.keras.regularizers.l2(reg)
        )(outputs)
        outputs.set_shape([None, None, 320, 128])
        outputs = bn_relu_fn(outputs)
        outputs = tf.keras.layers.Dropout(
            name=get_name_scope() + 'dropout',
            rate=.2
        )(outputs)

    with tf.name_scope('output'):

        with tf.name_scope('fusion'):
            outputs = tf.keras.layers.Dense(
                name=get_name_scope() + 'dense',
                use_bias=False,
                units=64,
                activation=None
            )(outputs)
            outputs = bn_relu_fn(outputs)
            outputs = tf.keras.layers.Dropout(
                name=get_name_scope() + 'dropout',
                rate=.2
            )(outputs)

        with tf.name_scope('output'):
            outputs = tf.keras.layers.Dense(
                name=get_name_scope() + 'dense',
                use_bias=True,
                units=1,
                activation=None
            )(outputs)
            outputs.set_shape([1, None, 320, 1])
            outputs = tf.squeeze(outputs, axis=[0, -1])

    model = tf.keras.Model(inputs, outputs, name='shaun vocal')

    return model


if __name__ == '__main__':

    model = create_acoustic_model_fn()
    model.summary(line_length=150)

    for idx, w in enumerate(model.trainable_variables):
        print(idx, w.name, w.shape, w.device)

    inputs = tf.random.normal([1, 1200, 500])
    outputs = model(inputs)
    print(outputs.shape)










