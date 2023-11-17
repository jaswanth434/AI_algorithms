import os
import shutil

source_folder = './Images/'  # Replace with your source folder path
destination_folder = './Dataset/dogs/'  # The folder where you want to copy images

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for subdir, dirs, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):  # Add other file extensions if needed
            shutil.copy2(os.path.join(subdir, file), destination_folder)

print("Images have been copied to", destination_folder)
