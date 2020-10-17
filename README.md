# Image-Classification
Image Classification using Keras on CIFAR-10 dataset

Image classification is a fascinating deep learning project. Specifically, image classification comes under the computer vision project category.
In this project, we will build a convolution neural network in Keras with python on a CIFAR-10 dataset. First, we will explore our dataset, and then we will train our neural network using python and Keras.

# What is Image Classification
 - The classification problem is to categorize all the pixels of a digital image into one of the defined classes.
 - Image classification is the most critical use case in digital image analysis.
 - Image classification is an application of both supervised classification and unsupervised classification.
   - In supervised classification, we select samples for each target class. We train our neural      network on these target class samples and then classify new samples.
   - In unsupervised classification, we group the sample images into clusters of images having similar properties. Then, we classify each cluster into our intended classes.

# About Image Classification Dataset
CIFAR-10 is a very popular computer vision dataset. This dataset is well studied in many types of deep learning research for object recognition.

This dataset consists of 60,000 images divided into 10 target classes, with each category containing 6000 images of shape 32*32. This dataset contains images of low resolution (32*32), which allows researchers to try new algorithms. The 10 different classes of this dataset are:

1.   Airplane
2. Car
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

CIFAR-10 dataset is already available in the datasets module of Keras. We do not need to download it; we can directly import it from keras.datasets.

# Project Prerequisites:
The prerequisite to develop and execute image classification project is [Keras and Tensorflow installation](https://data-flair.training/blogs/install-keras-on-linux-windows/).

# Steps
All the required steps can be referred from source.py or Image_Classification_Cifar-10.ipynb file.

# Results
The model acheives an accuracy of 84% in just 15 epochs and by visualizing the Classifiaction Accuracy graph it can be estimated that the model can acheive about 90% accuracy if it is trained for 50-60 epochs.
