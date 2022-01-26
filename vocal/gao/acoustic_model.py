"""
Ping Gao Model
"""

import tensorflow as tf
from self_defined import get_name_scope


def create_acoustic_model_fn():

    num_freq_bins = 256
    inputs_cfp = tf.keras.Input([None, num_freq_bins, 3], batch_size=1, name='cfp')
    outputs = inputs_cfp

    for layer_idx in range(5):

        with tf.name_scope('block_{}'.format(layer_idx)):

            inputs = outputs

            outputs = tf.keras.layers.BatchNormalization(
                name=get_name_scope() + 'bn',
                scale=False,
                center=False
            )(outputs)

            dilation = [3, 6, 12, 18, 24][layer_idx]
            outputs = tf.keras.layers.Conv2D(
                name=get_name_scope() + 'conv',
                kernel_size=[3, 3],
                dilation_rate=[dilation, dilation],
                padding='SAME',
                use_bias=False,
                filters=10,
                activation='selu',
                kernel_initializer='lecun_normal'
            )(outputs)
            outputs.set_shape([1, None, num_freq_bins, 10])
            outputs = tf.concat([inputs, outputs], axis=-1)
            outputs.set_shape([1, None, num_freq_bins, 3 + (layer_idx + 1) * 10])

    inputs = tf.tile(inputs_cfp, [1, 1, 1, 8])
    inputs.set_shape([1, None, num_freq_bins, 24])
    outputs = tf.concat([outputs, inputs], axis=-1)
    outputs.set_shape([1, None, num_freq_bins, 77])

    with tf.name_scope('pitch'):

        for layer_idx in range(3):
            with tf.name_scope('block_{}'.format(layer_idx)):

                outputs = tf.keras.layers.BatchNormalization(
                    name=get_name_scope() + 'bn',
                    scale=False,
                    center=False
                )(outputs)
                outputs = tf.keras.layers.Conv2D(
                    name=get_name_scope() + 'conv',
                    kernel_size=[3, 3],
                    padding='SAME',
                    use_bias=False,
                    activation='selu',
                    kernel_initializer='lecun_normal',
                    filters=[64, 32, 1][layer_idx]
                )(outputs)

                if layer_idx == 0:
                    features_for_voicing = outputs
        outputs.set_shape([1, None, 256, 1])
        outputs = tf.squeeze(outputs, axis=[0, -1])
        pitch_outputs = outputs

    with tf.name_scope('voicing'):

        outputs = features_for_voicing
        outputs = tf.keras.layers.BatchNormalization(
            name=get_name_scope() + 'bn',
            scale=False,
            center=False
        )(outputs)
        outputs = tf.keras.layers.Conv2D(
            name=get_name_scope() + 'conv',
            kernel_size=[3, 3],
            use_bias=False,
            padding='SAME',
            activation='selu',
            kernel_initializer='lecun_normal',
            filters=1
        )(outputs)
        outputs.set_shape([1, None, num_freq_bins, 1])
        outputs = tf.reduce_mean(outputs, axis=2)
        outputs = tf.squeeze(outputs, axis=[0, -1])
        voicing_outputs = outputs

    outputs = tf.concat([voicing_outputs[:, None], pitch_outputs], axis=-1)
    outputs.set_shape([None, 1 + num_freq_bins])

    model = tf.keras.Model(inputs_cfp, outputs, name='ping')

    return model


if __name__ == '__main__':
    model = create_acoustic_model_fn()
    model.summary(line_length=150)

    for idx, w in enumerate(model.trainable_variables):
        print(idx, w.name, w.shape)

    inputs = tf.random.uniform(shape=[1, 750, 256, 3])
    outputs = model(inputs)
    print(outputs.shape)


















