"""
w/ considering special requirements imposed by selu
"""

import tensorflow as tf
from self_defined import get_name_scope


class UnPooling(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(UnPooling, self).__init__(**kwargs)

    def call(self, inputs, indices, shape_before):
        updates = tf.reshape(inputs, [-1])
        indices = tf.reshape(indices, [-1, 1])
        size_before = tf.reduce_prod(shape_before)
        outputs = tf.scatter_nd(indices=indices, shape=[size_before], updates=updates)
        outputs = tf.reshape(outputs, shape_before)

        return outputs


def create_acoustic_model_fn():
    num_freq_bins = 320
    inputs = tf.keras.Input([None, num_freq_bins, 3], batch_size=1, name='cfp')

    outputs = inputs

    with tf.name_scope('encoder'):
        argmax_list = []
        for layer_idx in range(3):
            with tf.name_scope('layer_{}'.format(layer_idx)):
                outputs = tf.keras.layers.BatchNormalization(
                    name=get_name_scope() + 'bn',
                    scale=False,
                    center=False
                )(outputs)

                outputs = tf.keras.layers.Conv2D(
                    name=get_name_scope() + 'conv',
                    kernel_size=[5, 5],
                    padding='SAME',
                    use_bias=False,
                    filters=[32, 64, 128][layer_idx],
                    kernel_initializer='lecun_normal',
                    activation='selu'
                )(outputs)

                pooling_size = [1, 1, 4, 1]

                output_shape = tf.shape(outputs)
                outputs, argmax = tf.nn.max_pool_with_argmax(
                    name=get_name_scope() + 'maxpool',
                    input=outputs,
                    ksize=pooling_size,
                    strides=pooling_size,
                    padding='VALID',
                    output_dtype=tf.int64,
                    include_batch_in_index=True
                )
                argmax_list.append(dict(shape=output_shape, argmax=argmax))
        encoder_outputs = outputs
        encoder_outputs.set_shape([1, None, num_freq_bins // 64, 128])

    with tf.name_scope('non_melody'):

        outputs = tf.keras.layers.BatchNormalization(
            name=get_name_scope() + 'bn',
            scale=False,
            center=True
        )(encoder_outputs)
        outputs = tf.pad(outputs, [[0, 0], [2, 2], [0, 0], [0, 0]])
        non_melody_outputs = tf.keras.layers.Conv2D(
            name=get_name_scope() + 'conv',
            kernel_size=[5, 5],
            padding='VALID',
            use_bias=True,
            filters=1
        )(outputs)
        non_melody_outputs.set_shape([1, None, 1, 1])

    with tf.name_scope('decoder'):

        outputs = encoder_outputs
        for layer_idx in (2, 1, 0):
            with tf.name_scope('layer_{}'.format(layer_idx)):

                t = argmax_list[layer_idx]
                shape_before = t['shape']
                argmax = t['argmax']
                outputs = UnPooling(
                    name=get_name_scope() + 'unpooling'
                )(inputs=outputs, indices=argmax, shape_before=shape_before)

                if layer_idx > 0:

                    outputs = tf.keras.layers.BatchNormalization(
                        name=get_name_scope() + 'bn',
                        scale=False,
                        center=False
                    )(outputs)

                    outputs = tf.keras.layers.Conv2D(
                        name=get_name_scope() + 'conv',
                        kernel_size=[5, 5],
                        padding='SAME',
                        use_bias=False,
                        filters=[1, 32, 64][layer_idx],
                        kernel_initializer='lecun_normal',
                        activation='selu'
                    )(outputs)
                else:

                    outputs = tf.keras.layers.BatchNormalization(
                        name=get_name_scope() + 'bn',
                        scale=False,
                        center=True
                    )(outputs)

                    outputs = tf.keras.layers.Conv2D(
                        name=get_name_scope() + 'conv',
                        kernel_size=[5, 5],
                        padding='SAME',
                        use_bias=True,
                        filters=1
                    )(outputs)
        decoder_outputs = outputs
        decoder_outputs.set_shape([1, None, num_freq_bins, 1])

        combined = tf.concat([non_melody_outputs, decoder_outputs], axis=-2)
        combined.set_shape([1, None, num_freq_bins + 1, 1])
        combined = tf.squeeze(combined, axis=[0, -1])
        combined.set_shape([None, num_freq_bins + 1])

    model = tf.keras.Model(inputs, combined, name='hsieh correct')

    return model


if __name__ == '__main__':

    model = create_acoustic_model_fn()
    model.summary(line_length=150)

    for idx, w in enumerate(model.trainable_variables):
        print(idx, w.name, w.shape)

    inputs = tf.random.normal([1, 430, 360, 3])
    outputs = model(inputs, training=True)
    print(outputs.shape)




