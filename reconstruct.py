import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Add, GlobalAveragePooling2D, Multiply, Dense, Reshape, UpSampling2D, AveragePooling2D # type: ignore
from keras._tf_keras.keras.layers import Conv2D, Add, GlobalAveragePooling2D, Multiply, Dense, Reshape, UpSampling2D, AveragePooling2D 


class SqueezeExcitationLayer(tf.keras.layers.Layer):
    def __init__(self, input_channel, ratio=16, **kwargs):
        super(SqueezeExcitationLayer, self).__init__(**kwargs)
        self.reduction_ratio = ratio
        self.input_channels = input_channel

    def build(self, input_shape):
        self.global_av_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(self.input_channels // self.reduction_ratio, activation='relu')
        self.dense2 = Dense(self.input_channels, activation='sigmoid')
        super(SqueezeExcitationLayer, self).build(input_shape)

    def call(self, inputs):
        se = self.global_av_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = Reshape((1, 1, self.input_channels))(se)
        return Multiply()([se, inputs])

def reduction_conv(input_tensor, filters):
    x = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    x = SqueezeExcitationLayer(filters)(x)
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    x = Add()([x, residual])
    return x

def four_times(input_tensor, filters):
    x = input_tensor
    for _ in range(4):
        x = reduction_conv(x, filters)
    x = SqueezeExcitationLayer(filters)(x)
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    x = Add()([x, residual])
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    return x

def three_times(input_tensor, filters):
    x = input_tensor
    for _ in range(3):
        x = reduction_conv(x, filters)
    x = SqueezeExcitationLayer(filters)(x)
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    x = Add()([x, residual])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    return x

def two_times(input_tensor, filters):
    x = input_tensor
    for _ in range(2):
        x = reduction_conv(x, filters)
    x = SqueezeExcitationLayer(filters)(x)
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    x = Add()([x, residual])
    for _ in range(2):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    return x

def one_time(input_tensor, filters):
    x = input_tensor
    x = reduction_conv(x, filters)
    x = SqueezeExcitationLayer(filters)(x)
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    x = Add()([x, residual])
    for _ in range(3):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    return x

def zero_time(input_tensor, filters):
    x = input_tensor
    x = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    x = SqueezeExcitationLayer(filters)(x)
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)
    x = Add()([x, residual])
    for _ in range(4):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    return x

def create_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x_h2 = AveragePooling2D(2)(inputs)
    x_h4 = AveragePooling2D(2)(x_h2)
    x_h8 = AveragePooling2D(2)(x_h4)
    x_h16 = AveragePooling2D(2)(x_h8)
    x_h32 = AveragePooling2D(2)(x_h16)

    features1 = four_times(x_h2, 64)
    features2 = three_times(x_h4, 128)
    features3 = two_times(x_h8, 256)
    features4 = one_time(x_h16, 512)
    features5 = zero_time(x_h32, 512)

    # Aligning the dimensions for addition
    features2 = Conv2D(64, (1, 1), padding='same', activation='relu')(features2)
    features3 = Conv2D(64, (1, 1), padding='same', activation='relu')(features3)
    features4 = Conv2D(64, (1, 1), padding='same', activation='relu')(features4)
    features5 = Conv2D(64, (1, 1), padding='same', activation='relu')(features5)

    block1 = Add()([features1, features2])
    block2 = Add()([block1, features3])
    block3 = Add()([block2, features4])
    block4 = Add()([block3, features5])

    output = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(block4)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model

inputs = tf.keras.Input((256, 256, 3))
model = create_model(inputs.shape[1:])
model.summary()
