import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, add, Dropout, Reshape
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, concatenate, Lambda
import time


def ResNet_Block(input,block_id,filterNum):
    ''' Create a ResNet block
    Args:
        input: input tensor
        filterNum: number of output filters
    Returns: a keras tensor
    '''
    x = BatchNormalization()(input)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((1, 4))(x)

    init = Conv2D(filterNum, (1, 1), name='conv'+str(block_id)+'_1x1', padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = Conv2D(filterNum, (3, 3), name='conv'+str(block_id)+'_1', padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv2D(filterNum, (3, 3),  name='conv'+str(block_id)+'_2', padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    x = add([init, x])

    return x


def create_acoustic_model_fn():

    input = tf.keras.Input([31, 513], name='stft')

    block_1 = Conv2D(64, (3, 3), name='conv1_1', padding='same', kernel_initializer='he_normal', use_bias=False,
                         kernel_regularizer=l2(1e-5))(input[..., None])
    block_1 = BatchNormalization()(block_1)
    block_1 = LeakyReLU(0.01)(block_1)
    block_1 = Conv2D(64, (3, 3), name='conv1_2', padding='same', kernel_initializer='he_normal', use_bias=False,
                     kernel_regularizer=l2(1e-5))(block_1)

    block_2 = ResNet_Block(input=block_1, block_id=2, filterNum=128)
    block_3 = ResNet_Block(input=block_2, block_id=3, filterNum=192)
    block_4 = ResNet_Block(input=block_3, block_id=4, filterNum=256)
    block_4.set_shape([None, 31, 8, 256])

    block_4 = BatchNormalization()(block_4)
    block_4 = LeakyReLU(0.01)(block_4)
    block_4 = MaxPooling2D((1, 4))(block_4)
    block_4 = Dropout(0.5)(block_4)
    block_4.set_shape([None, 31, 2, 256])

    pitch = Reshape((31, 512))(block_4)

    pitch = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.3, dropout=0.3, unroll=True))(pitch)
    pitch = Dense(722, activation=None, name='pitch')(pitch)

    block_1 = MaxPooling2D((1, 4 ** 4))(block_1)
    block_2 = MaxPooling2D((1, 4 ** 3))(block_2)
    block_3 = MaxPooling2D((1, 4 ** 2))(block_3)
    block_3.set_shape([None, 31, 2, None])

    voicing = concatenate([block_1, block_2, block_3, block_4])
    voicing = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(1e-5))(voicing)
    voicing = BatchNormalization()(voicing)
    voicing = LeakyReLU(0.01)(voicing)
    voicing = Dropout(0.5)(voicing)

    voicing = Reshape((31, 512))(voicing)
    voicing = Bidirectional(LSTM(32, return_sequences=True, stateful=False, recurrent_dropout=0.3, dropout=0.3, unroll=True))(
            voicing)
    voicing = Dense(2, activation='softmax')(voicing)

    pitch_n_voicing = tf.exp(pitch)
    t = tf.reduce_sum(pitch_n_voicing, axis=-1)
    pitch_n_voicing = pitch_n_voicing[..., 0] / t
    pitch_voicing = -pitch_n_voicing + 1.
    pitch_voicing = tf.stack([pitch_n_voicing, pitch_voicing], axis=-1)
    voicing = voicing + pitch_voicing
    n_voicing, voicing = tf.unstack(voicing, axis=-1)
    voicing = voicing - n_voicing

    model = tf.keras.Model(input, outputs=dict(pitch=pitch, voicing=voicing))

    return model


if __name__ == '__main__':

    model = create_acoustic_model_fn()

    batch_size = 64
    inputs = tf.random.uniform([batch_size, 31, 513])
    outputs = model(inputs, training=True)
    reg_losses = model.losses
    print()




