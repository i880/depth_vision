import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Assuming the create_model function is defined as shown previously

# Define paths to your training and validation data directories
train_dir = '/path/to/train'
val_dir = '/path/to/val'

# Image data generator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Rescale pixel values to [0, 1]
    shear_range=0.2,         # Shear intensity
    zoom_range=0.2,          # Zoom range
    horizontal_flip=True     # Randomly flip images horizontally
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale validation data

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',  # Assuming we are performing image-to-image translation
    color_mode='rgb'     # Load RGB images
)

# Load and preprocess validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',  # Assuming we are performing image-to-image translation
    color_mode='rgb'     # Load RGB images
)

# Modify the model's output layer to match depth map format (single-channel)
def create_depth_model(input_shape):
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

    output = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(block4)  # Single-channel output

    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model

# Compile the model for depth estimation
model = create_depth_model((256, 256, 3))
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',  # Suitable loss function for depth estimation
    metrics=['mean_absolute_error']  # Example metric for evaluation
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=50  # Number of epochs to train
)

# Save the model
model.save('depth_estimation_model.h5')

# Load the model (if needed)
# model = tf.keras.models.load_model('depth_estimation_model.h5')
