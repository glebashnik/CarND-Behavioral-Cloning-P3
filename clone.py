import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Lambda, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from PIL import Image

data_path = 'data/driving_log.csv'
img_dir_path = 'data/IMG/'

images = []
steering = []

def load_image(orig_img_path):
    img_file_name = orig_img_path.split('/')[-1]
    img_path = img_dir_path + img_file_name
    return np.asarray(Image.open(img_path))

with open(data_path, 'r') as f:
    reader = csv.reader(f)
    
    for row in reader:
        steering_center = float(row[3])

        correction = 0.1
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        img_center = load_image(row[0])
        img_left = load_image(row[1])
        img_right = load_image(row[2])

        images.extend([img_center, img_left, img_right])
        steering.extend([steering_center, steering_left, steering_right])

# aug_images, aug_measurements = [], []

# for image, measurement in zip(images, measurements):
#     aug_images.append(image)
#     aug_measurements.append(measurement)

#     aug_images.append(cv2.flip(image, 1))
#     aug_measurements.append(measurement * -1.0)

X_train = np.array(images)
y_train = np.array(steering)

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D((4, 4)))
model.add(Conv2D(8, (5, 5), activation='relu'))
model.add(MaxPooling2D((4, 4)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('model2.h5')
exit()