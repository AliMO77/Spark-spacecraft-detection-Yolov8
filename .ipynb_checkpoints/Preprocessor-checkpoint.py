# MyImageProcessor.py

import sys
import cv2
import random

import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
import os 
import shutil


class Preprocessor:

    def __init__(self,train_image_folder,val_image_folder,train_data_csv,val_data_csv):
        self.train_image_folder = train_image_folder
        self.val_image_folder = val_image_folder
        self.train_data_csv = train_data_csv
        self.val_data_csv = val_data_csv
        self.train_df, self.val_df = self.load_and_process_csv_data()
        self.all_labels = self.get_possible_labels()
        self.one_hot_encode_labels()
        self.make_numeric_class(self.train_df)
        self.make_numeric_class(self.val_df)

       
    
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
    
    def make_numeric_class(self,dataframe):
        num_labels = []
        for index, row in dataframe.iterrows():
            idx = np.where( self.all_labels== row['class'])[0][0]
            num_labels.append(idx)

        dataframe['num_labels'] = num_labels

    def process_bbox_string(self, bbox_str):
        bbox_coords = [int(coord) for coord in bbox_str.strip('[]').split(',')]
        return np.array(bbox_coords)


    def select_classes(self, selected_classes):
    
        filtered_training_df = self.train_df[self.train_df['class'].isin(selected_classes)]

        filtered_val_df = self.val_df[self.val_df['class'].isin(selected_classes)]
        
        return filtered_training_df, filtered_val_df
    
    def make_train_val_dfs(self,df, train_size=0.8, random_state=42):
   
        # Step 2: Shuffle the DataFrame
        df = df.sample(frac=1, random_state=random_state)

        # Step 3: Split into train and validation sets
        train_df, val_df = train_test_split(df, train_size=train_size, random_state=random_state)

        # Ensure that both train and val sets have all unique classes
        for unique_class in self.all_labels:
            if unique_class not in train_df['class'].unique():
                additional_samples = val_df[val_df['class'] == unique_class].sample(frac=0.5, random_state=random_state)
                train_df = train_df.append(additional_samples)
                val_df = val_df.drop(additional_samples.index)

        return train_df, val_df
    
    def move_files_to_val_directory(self,valdf, source_directory, destination_directory):
        # Create the 'val' directory if it doesn't exist
        os.makedirs(destination_directory, exist_ok=True)

        # Iterate through the DataFrame and move files to the 'val' directory
        for index, row in valdf.iterrows():
            source_path = os.path.join(source_directory, row['filename'])
            destination_path = os.path.join(destination_directory, row['filename'])
            
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
            else:
                print(f"File not found: {source_path}")

    
    def Yolo_labels_maker(self,dataframe,output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        for index, row in dataframe.iterrows():
            # Replace the extension with ".txt"
            filename = os.path.join(output_dir, os.path.splitext(row['filename'])[0] + ".txt")
            with open(filename, 'w') as file:
                bbox = [float(value / 1024) for value in row['bbox']]
                y1,x1,y2,x2 = bbox
                file.write(f"{row['num_labels']} {(y1)} {(x1)} {(y2)} {(x2)}")



  #####################################################################################################################################  
   

