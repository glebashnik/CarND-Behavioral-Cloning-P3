# Behavioral Cloning Project

## Project goal

Train a model to drive a car around a track in a simulator based on the data collected from manual driving.

## Project plan

Steps of this project are the following:

1. Use the simulator to collect data of good driving behavior
1. Build, a convolution neural network in Keras that predicts steering angles from images
1. Train and validate the model with a training and validation set
1. Test that the model successfully drives around track one without leaving the road
1. Summarize the results with a written report

[losses]: ./examples/losses.png "Train and validation losses"
[original]: ./examples/original.jpg "Original image"
[flipped]: ./examples/flipped.jpg "Flipped image"
[left]: ./examples/left.jpg "Left image"
[center]: ./examples/center.jpg "Center image"
[right]: ./examples/right.jpg "Right image"
[cropped]: ./examples/cropped.jpg "Cropped image"

## Project files

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup_report.md` summarizing the results

To train and save a model in `model.h5`:

```sh
python model.py
```

To test the model, in a simulator provided by Udacity:

```sh
python drive.py model.h5
```

## Model architecture

The model takes 320x160 RGB image as an input and returns one value, corresponding to a steering wheel angle.
I used a Lenet-based architecture with dropout before the output layer to reduce overfitting (function `lenet`).

| Layer             | Output Shape         | Param #   |
|-------------------|----------------------|-----------|
| Cropping          | (None, 90, 320, 3)   | 0         |
| Normalization     | (None, 90, 320, 3)   | 0         |
| Conv2D            | (None, 86, 316, 16)  | 1216      |
| MaxPooling2       | (None, 21, 79, 16)   | 0         |
| Conv2D            | (None, 17, 75, 8)    | 3208      |
| MaxPooling2       | (None, 4, 18, 8)     | 0         |
| Flatten           | (None, 576)          | 0         |
| Dense             | (None, 120)          | 69240     |
| Dense             | (None, 84)           | 10164     |
| Dropout           | (None, 84)           | 0         |
| Output            | (None, 1)            | 85        |

Total params: 83,913

I also implemented the model based on NVIDIA architecture (function `nvidia`). However, for the test track a Lenet-based architecture performs as well or better than NVIDIA-based one which has 10 times more parameters thus higher chance to overfit and requires more data for training.

Lenet was the first model I tried and it performed well with my training data. Adding layers, changing layer sizes, number of filters in convolution layers didn't improve performance in a significant way.

## Model and training parameters

The model used an adam optimizer, so the learning rate was not tuned manually. Parameters that were left to tune:

* Batch size
* Drop probability for the dropout layer
* Number of epochs to train the model

Changing the batch size had a significant effect on the performance of the model. Training with a batch size larger than 10 might result in the model that can't handle *unusual* parts of the track, e.g. no lane marking on the right side, bridge, sharp turn. This can be explained by averaging of the gradient in the batch, thus the model will not be able to adapt to unusual, short parts of the track. At the end I settled on the batch size of 4.

Drop probability was set to 0.4, a small decrease from standart 0.5, which achives lower loss.

The model was trained for 7 epochs, which was decided by looking at training and validation error. Training for more than 7 epochs increases validation error due to overfitting.

![Train and validation losses][losses]

## Training data

### Generating training data

Training data was generation by driving two laps on the first track manually with a keyboard. I kept the car in the middle of the road with minor deviations in sharp turns. This data was enough to train the model to keep the vehicle on the road with relatively smooth steering. No recovery training data was needed. In total, my training data contained 2463 points.

### Adding side images

Each training data point contains three images: left, center, right.

Left

![alt text][left]

Center

![alt text][center]

Right

![alt text][right]

These images are added as independent points with steering angle correction for left (+ 0.1) and right (-0.1) images (function `load_data`). Angle correction was tweaked manually to minimize the validation error and test performance on the track.

### Adding flipped images

To extend the training data I added points where the images are flipped horisontally and the steering angle is multiplied by `-1`.

Original image

![alt text][original]

Flipped image

![alt text][flipped]

### Cropping images

To simplify the task, top and bottom part of images were removed because it is less relevant for determining the steering angle (part of the model function `lenet` and `nvidia`)

Original image

![alt text][original]

Cropped image

![alt text][cropped]

## Implementaion details

Loading and processing of training data was implemented using generators to avoid unncessary memory consumption (function `data_generator`). Training samples are shuffled and splitted into a training set 80% and testing set 20% (function `main`).

The code is organised into functions documented with comments, making it easier to understand and reuse the code.

I used Keras 2.0.5, to get used to changes in the API.

## Result

The model was tested in a simulator on a first track. The resulting video is included in [video.mp4](video.mp4).
As can be seen in the video, the car is mostly centered and the steering is smooth.