# Gesture-Recognition

# Problem Statement:
Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

Thumbs up: Increase the volume
Thumbs down: Decrease the volume
Left swipe: 'Jump' backwards 10 seconds
Right swipe: 'Jump' forward 10 seconds
Stop: Pause the movie
Each video is a sequence of 30 frames (or images)

# Understanding the Dataset
The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use.


# Two Architectures: 3D Convs and CNN-RNN Stack
After understanding and acquiring the dataset, the next step is to try out different architectures to solve this problem.

For analysing videos using neural networks, two types of architectures are used commonly.

One is the standard CNN + RNN architecture in which you pass the images of a video through a CNN which extracts a feature vector for each image, and then pass the sequence of these feature vectors through an RNN.


# Data Preprocessing
We can apply several of the image procesing techniques for each of image in the frame.

# Crop, Resize and Normalization
Images are of different sizes hence converted each image of the train and test set into a matrix of size 120*120 by croping, Normalization and resizing.

# Generators
Understanding Generators: As you already know, in most deep learning projects you need to feed data to the model in batches. This is done using the concept of generators.

Creating data generators is probably the most important part of building a training pipeline. Although libraries such as Keras provide builtin generator functionalities, they are often restricted in scope and you have to write your own generators from scratch. In this project we will implement our own cutom generator, our generator will feed batches of videos, not images.

Let's take an example, assume we have 23 samples and we pick batch size as 10.

In this case there will be 2 complete batches of ten each

Batch 1: 10
Batch 2: 10
Batch 3: 3
The final run will be for the remaining batch that was not part of the the full batch.


Implementation
3D Convolutional Network, or Conv3D
Now, lets implement a 3D convolutional Neural network on this dataset. To use 2D convolutions, we first convert every image into a 3D shape : width, height, channels. Channels represents the slices of Red, Green, and Blue layers. So it is set as 3. In the similar manner, we will convert the input dataset into 4D shape in order to use 3D convolution for : length, breadth, height, channel (r/g/b).

The architecture is described below:

While I tried with multiple filter size, bigger filter size is resource intensive and we have done most experiment with 3*3 filter. I have used sgd(Stochastic gradient descent) optimizer with its default settings. We have additionally used the ReduceLROnPlateau metrics to reduce the learning rate.

