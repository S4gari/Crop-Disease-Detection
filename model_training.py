import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

img_size = 128
batch_size = 32
train_path = 'C:/Users/sagar/OneDrive/Pictures/Documents/Desktop/CropDiseaseDetection/dataset/train'
test_path = 'C:/Users/sagar/OneDrive/Pictures/Documents/Desktop/CropDiseaseDetection/dataset/test'

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(train_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')
test = test_datagen.flow_from_directory(test_path, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train, validation_data=test, epochs=10)

model.save('crop_disease_model.h5')
