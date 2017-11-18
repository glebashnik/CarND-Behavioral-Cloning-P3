import csv
import random
from PIL import Image
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Lambda, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

def load_log(log_path):
    """Read log rows into list."""
    rows = []

    with open(log_path, 'r') as log_file:
        reader = csv.reader(log_file)

        for row in reader:
            rows.append(row)

    return rows

def load_image(orig_path, dir_path):
    """Loads an image, replacing the original path with the one inside the project."""
    file_name = orig_path.split('/')[-1]
    path = dir_path + file_name
    return np.asarray(Image.open(path))

def load_data(log_rows, img_dir_path):
    """Loads center, left, right images nad a steering angle for each row in the log."""
    images = []
    angles = []
    
    for row in log_rows:
        image_center = load_image(row[0], img_dir_path)
        image_left = load_image(row[1], img_dir_path)
        image_right = load_image(row[2], img_dir_path)

        angle_center = float(row[3])
        angle_correction = 0.1
        angle_left = angle_center + angle_correction
        angle_right = angle_center - angle_correction

        images.extend([image_center, image_left, image_right])    
        angles.extend([angle_center, angle_left, angle_right])

    return  images, angles

def flip(images, angles):
    """Flips an image horisontally."""
    flipped_images, flipped_angles = [], []

    for image, angle in zip(images, angles):
        flipped_images.append(image)
        flipped_angles.append(angle)

        flipped_images.append(cv2.flip(image, 1))
        flipped_angles.append(angle * -1.0)

    return flipped_images, flipped_angles

def data_generator(log_rows, img_dir_path, batch_size=32):
    """Creates a generator for training data, includes side and flipped images for each sample."""
    num_rows = len(log_rows)

    while 1:
        random.shuffle(log_rows)

        for offset in range(0, num_rows, batch_size):
            batch_rows = log_rows[offset:offset+batch_size]
            images, angles = load_data(batch_rows, img_dir_path)
            aug_images, aug_angles = flip(images, angles)
            X = np.array(aug_images)
            y = np.array(aug_angles)
            yield sklearn.utils.shuffle(X, y)

def lenet(drop_prob):
    """Implementation of Lenet-based model with dropout layer before the output layer."""
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
    model.add(Dropout(drop_prob))
    model.add(Dense(1))
    return model

def nvidia(drop_prob):
    """Implementation of NVIDIA-based model with dropout layer before the output layer."""
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(drop_prob))
    model.add(Dense(1))
    return model

def plot_losses(history_obj):
    """Plots training and validation losses from the Keras history object."""
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def main():
    # Paths
    log_path = 'data/train1/driving_log.csv'
    img_dir_path = 'data/train1/IMG/'
    
    # Parameters for manual tuning
    validation_size = 0.2
    batch_size = 4
    epochs = 7
    drop_prob = 0.4

    # Loading data and creating training and validation generators.
    samples = load_log(log_path)
    train_samples, validation_samples = train_test_split(samples, test_size=validation_size)
    train_generator = data_generator(train_samples, img_dir_path, batch_size=batch_size)
    validation_generator = data_generator(validation_samples, img_dir_path, batch_size=batch_size)

    # Creating and training a model.
    model = lenet(drop_prob=drop_prob)
    model.summary()
    model.compile(loss='mse', optimizer='adam')

    history_obj = model.fit_generator(
        generator=train_generator, 
        steps_per_epoch=len(train_samples) / batch_size + 1, 
        validation_data=validation_generator,
        validation_steps=len(validation_samples) / batch_size + 1,
        epochs=epochs
    )

    # Saving the model, plotting losses
    #model.save('model.h5')
    plot_losses(history_obj)

if __name__ == "__main__":
    main()