# MyImageProcessor.py

import sys
import cv2
import random

import numpy as np

import tensorflow as tf
from tensorflow import keras


from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd
import os 

class ImageProcessor:

    def __init__(self,train_image_folder,val_image_folder,train_data_csv,val_data_csv):
        self.train_image_folder = train_image_folder
        self.val_image_folder = val_image_folder
        self.train_data_csv = train_data_csv
        self.val_data_csv = val_data_csv
        self.train_df, self.val_df = self.load_and_process_csv_data()
        self.all_labels = self.get_possible_labels()
        self.one_hot_encode_labels()
       
    
    def load_and_process_csv_data(self):
        train_data_pd = pd.read_csv(self.train_data_csv)
        val_data_pd = pd.read_csv( self.val_data_csv )
        
        
        train_data_pd['bbox']= train_data_pd['bbox'].apply(self.process_bbox_string)
        train_data_pd['filename'] =  train_data_pd['filename'].str.replace('png', 'jpg')
        train_data_pd['filepath'] = self.train_image_folder + train_data_pd['filename']

        val_data_pd['bbox']= val_data_pd['bbox'].apply(self.process_bbox_string)
        val_data_pd['filename'] =  val_data_pd['filename'].str.replace('png', 'jpg')
        val_data_pd['filepath'] = self.val_image_folder + val_data_pd['filename']

        return train_data_pd,val_data_pd

    def get_possible_labels(self):
        all_labels = self.train_df['class'].unique()
        return all_labels
   
    def one_hot_encode_labels(self):
        def encode(df):
            encoded_labels = []
            for label in df['class']:
                one_hot = np.zeros(len(self.all_labels),dtype=np.float32)
                idx = np.where(self.all_labels == label )
                one_hot[idx] = 1
                encoded_labels.append(one_hot)
            df['encoded_labels'] = encoded_labels

        encode(self.train_df)
        encode(self.val_df)
    

    
  #####################################################################################################################################  
    def process_bbox_string(self, bbox_str):
        bbox_coords = [int(coord) for coord in bbox_str.strip('[]').split(',')]
        return bbox_coords


    def plot_images_with_bboxes(self, images, titles, image_shapes, bboxes_list, rows=2):
        num_images = len(images)
        num_cols = 2

        if num_images == 0:
            print("No images to display.")
            return

        num_rows = (num_images + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))

        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(num_images):
            ax = axes[i // num_cols, i % num_cols]
            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            
            bbox = bboxes_list[i]

            if isinstance(bbox, str):
                bbox = eval(bbox)
                
            y1, x1, y2, x2 = bbox
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ax.imshow(img_rgb)
            ax.set_title(f"class: {titles[i]} | Res:[({image_shapes[i][0]}x{image_shapes[i][1]}),{image_shapes[i][2]}]")
            ax.axis('off')

        for i in range(num_images, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        plt.show()

    def display_random_image_per_class(self,data_pd):

        class_names = data_pd['class'].unique().tolist()
        images = []
        image_shapes = []
        bboxes = []

        for label in class_names:
            selected_class = data_pd[data_pd['class'] == label]
            if len(selected_class) == 0:
                print(f"No images found for class '{label}'")
                continue

            random_index = random.randint(0, len(selected_class) - 1)
            random_image_row = selected_class.iloc[random_index]
            filepath = self.train_image_folder + random_image_row['filename']
            bboxes.append(random_image_row['bbox'])
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = image.shape
            images.append(image)
            image_shapes.append([height, width, channels])

        self.plot_images_with_bboxes(images, class_names, image_shapes, bboxes)
        
    


    def display_images(self, cropped_images, sample_size=None):
        """
        Display a sample of cropped images.

        Args:
            cropped_images (list): List of images (numpy arrays).
            sample_size (int, optional): Number of images to display. Default is None (displays all).

        Returns:
            None
        """
        num_images = len(cropped_images)

        if sample_size is not None:
            num_images = min(num_images, sample_size)

        if num_images == 0:
            print("No images to display.")
            return
        
            
        num_cols = 2  # Two images per row
        num_rows = (num_images + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))

        # Flatten axes if there's only one row
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(num_images):
            row_index = i // num_cols
            col_index = i % num_cols
            ax = axes[row_index, col_index]

            normalized_image = cropped_images[i]
            img = (normalized_image * 255).astype(np.uint8)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis('off')

        # Remove any empty subplots
        for i in range(num_images, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        plt.show()



    def plot_feature_distribution(self,df,feature):
        sns.set(style='whitegrid', rc={'axes.facecolor': '#ffffff','figure.facecolor': '#fdf8f7','axes.grid': False})
        
        plt.figure(figsize=(10,6))
        feature_counts = df[feature].value_counts()

        ax = sns.countplot(x=feature, data=df, palette='dark', order = feature_counts.index )
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Distribution of '+ feature, fontsize=16)
        
        sns.despine(left=True)
        plt.tick_params(labelsize=12)
        plt.xticks(rotation=45, ha='right')


        # Annotate each bar with its count
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
        plt.tight_layout()
        plt.show()
