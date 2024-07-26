import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Add, GlobalAveragePooling2D, Multiply, Dense, Reshape, UpSampling2D, AveragePooling2D # type: ignore
from keras._tf_keras.keras.layers import Conv2D, Add, GlobalAveragePooling2D, Multiply, Dense, Reshape, UpSampling2D, AveragePooling2D 

# Define a custom Squeeze and Excitation layer
class SqueezeExcitationLayer(tf.keras.layers.Layer):
    def __init__(self, input_channel, ratio=16, **kwargs):
        super(SqueezeExcitationLayer, self).__init__(**kwargs)
        self.reduction_ratio = ratio
        self.input_channels = input_channel

    def build(self, input_shape):
        self.global_av_pool = GlobalAveragePooling2D()  # Global average pooling layer
        self.dense1 = Dense(self.input_channels // self.reduction_ratio, activation='relu')  # First dense layer
        self.dense2 = Dense(self.input_channels, activation='sigmoid')  # Second dense layer
        super(SqueezeExcitationLayer, self).build(input_shape)

    def call(self, inputs):
        se = self.global_av_pool(inputs)  # Apply global average pooling
        se = self.dense1(se)  # Apply first dense layer
        se = self.dense2(se)  # Apply second dense layer
        se = Reshape((1, 1, self.input_channels))(se)  # Reshape for broadcasting
        return Multiply()([se, inputs])  # Multiply input with the SE output

# Function for a reduction convolution block with Squeeze and Excitation
def reduction_conv(input_tensor, filters):
    x = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)  # 1x1 convolution
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)  # 3x3 convolution
    x = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)  # 1x1 convolution
    x = SqueezeExcitationLayer(filters)(x)  # Apply SE layer
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)  # Residual connection
    x = Add()([x, residual])  # Add residual connection
    return x

# Function for four reduction convolution blocks with a final SE layer
def four_times(input_tensor, filters):
    x = input_tensor
    for _ in range(4):
        x = reduction_conv(x, filters)  # Apply reduction convolution 4 times
    x = SqueezeExcitationLayer(filters)(x)  # Apply final SE layer
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)  # Residual connection
    x = Add()([x, residual])  # Add residual connection
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)  # Final convolution
    return x

# Function for three reduction convolution blocks with upsampling
def three_times(input_tensor, filters):
    x = input_tensor
    for _ in range(3):
        x = reduction_conv(x, filters)  # Apply reduction convolution 3 times
    x = SqueezeExcitationLayer(filters)(x)  # Apply final SE layer
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)  # Residual connection
    x = Add()([x, residual])  # Add residual connection
    x = UpSampling2D(size=(2, 2))(x)  # Upsampling
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)  # Final convolution
    return x

# Function for two reduction convolution blocks with upsampling
def two_times(input_tensor, filters):
    x = input_tensor
    for _ in range(2):
        x = reduction_conv(x, filters)  # Apply reduction convolution 2 times
    x = SqueezeExcitationLayer(filters)(x)  # Apply final SE layer
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)  # Residual connection
    x = Add()([x, residual])  # Add residual connection
    for _ in range(2):
        x = UpSampling2D(size=(2, 2))(x)  # Upsampling twice
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)  # Final convolution
    return x

# Function for one reduction convolution block with upsampling
def one_time(input_tensor, filters):
    x = input_tensor
    x = reduction_conv(x, filters)  # Apply reduction convolution 1 time
    x = SqueezeExcitationLayer(filters)(x)  # Apply final SE layer
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)  # Residual connection
    x = Add()([x, residual])  # Add residual connection
    for _ in range(3):
        x = UpSampling2D(size=(2, 2))(x)  # Upsampling three times
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)  # Final convolution
    return x

# Function for zero reduction convolution block with upsampling
def zero_time(input_tensor, filters):
    x = input_tensor
    x = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)  # 1x1 convolution
    x = SqueezeExcitationLayer(filters)(x)  # Apply SE layer
    residual = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_tensor)  # Residual connection
    x = Add()([x, residual])  # Add residual connection
    for _ in range(4):
        x = UpSampling2D(size=(2, 2))(x)  # Upsampling four times
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)  # Final convolution
    return x

# Function to create the model
def create_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Downsampling with average pooling
    x_h2 = AveragePooling2D(2)(inputs)
    x_h4 = AveragePooling2D(2)(x_h2)
    x_h8 = AveragePooling2D(2)(x_h4)
    x_h16 = AveragePooling2D(2)(x_h8)
    x_h32 = AveragePooling2D(2)(x_h16)

    # Applying reduction convolutions and SE layers
    features1 = four_times(x_h2, 64)
    features2 = three_times(x_h4, 128)
    features3 = two_times(x_h8, 256)
    features4 = one_time(x_h16, 512)
    features5 = zero_time(x_h32, 512)

    # Aligning the dimensions for addition using 1x1 convolutions
    features2 = Conv2D(64, (1, 1), padding='same', activation='relu')(features2)
    features3 = Conv2D(64, (1, 1), padding='same', activation='relu')(features3)
    features4 = Conv2D(64, (1, 1), padding='same', activation='relu')(features4)
    features5 = Conv2D(64, (1, 1), padding='same', activation='relu')(features5)

    # Combining features
    block1 = Add()([features1, features2])
    block2 = Add()([block1, features3])
    block3 = Add()([block2, features4])
    block4 = Add()([block3, features5])

    # Final output layer
    output = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(block4)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model

# Define input shape
inputs = tf.keras.Input((256, 256, 3))
# Create and summarize the model
model = create_model(inputs.shape[1:])
model.summary()
