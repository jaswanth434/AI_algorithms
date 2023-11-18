import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def save_accuracy_loss_graphs(history, file_path_prefix):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r^-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(f'{file_path_prefix}_accuracy.png')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'bs-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'rv-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'{file_path_prefix}_loss.png')
    plt.close()


def predict_image_from_path_and_show(model, file_path):
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0

    predictions = model.predict(img_array)
    prediction_text = "house" if predictions[0][0] > 0.5 else "dog"

    plt.imshow(img)
    plt.title(f'Prediction: {prediction_text}')
    plt.axis('off')
    plt.show()

    return prediction_text


model_file_path = "./dog_or_house_model.h5"

if os.path.exists(model_file_path):
    
    model = tf.keras.models.load_model(model_file_path)

    while True:
        user_input = input("Enter a file path to predict (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            break
        try:
            prediction = predict_image_from_path_and_show(model, user_input)
            print("Prediction:", prediction)
        except Exception as e:
            print("Error:", e)
else:
        
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

    train_generator = train_datagen.flow_from_directory(
            './Dataset/training',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary',
            subset='training')

    validation_generator = train_datagen.flow_from_directory(
            './Dataset/training',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary',
            subset='validation')
            
    class_labels = list(train_generator.class_indices.keys())
    print(class_labels)

    total_images_per_class = {class_name: 0 for class_name in train_generator.class_indices.keys()}
    for _, labels in train_generator:
        for label in labels:
            class_name = list(train_generator.class_indices.keys())[int(label)]
            total_images_per_class[class_name] += 1
        if train_generator.batch_index == 0:
            break  

    print(total_images_per_class)

    model = create_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    callbacks = [early_stopping, reduce_lr]

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=14,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks)

    save_accuracy_loss_graphs(history, 'model_performance')
    model.save(model_file_path)



 
# ./Dataset/dog-puppy-on-garden-royalty-free-image-1586966191.jpg

# ./Dataset/interior-design-of-a-small-house.jpg

# ./Dataset/Contemporary-interior-design-and-decor-Urbanology-Designs.jpg

# ./Dataset/GettyImages-1285438779-2000-9ea25aa777df42e6a046b10d52b286b7.jpg

# ./Dataset/w575.jpg