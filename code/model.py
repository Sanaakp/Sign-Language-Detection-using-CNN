import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
train_path = 'dataset/train'
val_path = 'dataset/val'
test_path = 'dataset/test'

# Image Preprocessing
img_height, img_width = 64, 64
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    )
val_datagen = ImageDataGenerator(
    rescale=1./255,
    )
test_datagen = ImageDataGenerator(
    rescale=1./255,
    )

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Class names
class_names = list(train_generator.class_indices.keys())

# Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the model
model.save("asl_model.h5")
