import tensorflow as tf
import cv2
import numpy as np
import os 
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CustomImageDataGenerator:

    def __init__(self, dataframe, directory, batch_size, image_size=None, sample_size=None, shuffle=True):
        self.dataframe = dataframe
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.sample_size = sample_size

        if self.sample_size is not None:
            self.dataframe = self.dataframe.sample(self.sample_size)

        

        # Create a dataset from the dataframe
        self.data = tf.data.Dataset.from_tensor_slices((self.dataframe['filepath'],self.dataframe['encoded_labels'].tolist()))

        
        # Map the load_and_preprocess_image function to the dataset
        self.data = self.data.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Apply data augmentation
        self.data = self.data.map(self.data_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.data = self.data.repeat()

        # Shuffle and batch the dataset
        if self.shuffle:
            self.data = self.data.shuffle(buffer_size=1024)
        self.data = self.data.batch(self.batch_size)
        self.data = self.data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

   
   
       # Define a function to load and preprocess images
    def load_and_preprocess_image(self, image_path, labels):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        if self.image_size is not None:
            img = tf.image.resize(img, self.image_size)
        return img, labels

        
   
    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def data_augmentation(self, image, labels):
        # Define data augmentation operations here
        image = tf.image.random_flip_left_right(image)
        # Add more data augmentation as needed
        return image, labels
