import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
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
        Dense(1, activation='sigmoid')  # 1 for binary classification (dog or house)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)  # Splitting data into training and validation

train_generator = train_datagen.flow_from_directory(
        './Dataset/',  # Replace with the path to your dataset
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        './Dataset/',  # Replace with the path to your dataset
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation')
class_labels = list(train_generator.class_indices.keys())
print(class_labels)

# Count images per class
total_images_per_class = {class_name: 0 for class_name in train_generator.class_indices.keys()}
for _, labels in train_generator:
    for label in labels:
        class_name = list(train_generator.class_indices.keys())[int(label)]
        total_images_per_class[class_name] += 1
    if train_generator.batch_index == 0:
        break  # Stop after one complete pass through the dataset

print(total_images_per_class)
# exit()
# Create and train the model
model = create_model()
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples // train_generator.batch_size,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples // validation_generator.batch_size)

# Save the model
model.save('dog_or_house_model.h5')

# Function to predict an image from a URL
def predict_image_from_url(model, url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Rescale pixel values

    predictions = model.predict(img_array)
    if predictions[0] > 0.5:
        return "dog"
    else:
        return "house"

# Example usage
model = tf.keras.models.load_model('dog_or_house_model.h5')
url = 'https://thumbs.dreamstime.com/b/golden-retriever-dog-21668976.jpg'
prediction = predict_image_from_url(model, url)
print(prediction)
print("------")

url = 'https://thumbor.forbes.com/thumbor/fit-in/900x510/https://www.forbes.com/home-improvement/wp-content/uploads/2022/07/download-23.jpg'
prediction = predict_image_from_url(model, url)
print(prediction)
