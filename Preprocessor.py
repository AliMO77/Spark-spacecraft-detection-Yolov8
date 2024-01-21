# MyImageProcessor.py

import sys
import cv2
import random
import matplotlib.pyplot as plt 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import shutil
from PIL import Image
import matplotlib.patches as patches


class Preprocessor:

    def __init__(self,train_image_folder=None,val_image_folder=None,train_data_csv=None,val_data_csv=None):
       
        if train_image_folder is not None and val_image_folder is not None and   train_data_csv is not None and val_data_csv is not None :

            
            self.train_image_folder = train_image_folder
            self.val_image_folder = val_image_folder
            self.train_data_csv = train_data_csv
            self.val_data_csv = val_data_csv
            self.train_df, self.val_df = self.load_and_process_csv_data()
            self.all_labels = self.get_possible_labels()
            self.one_hot_encode_labels()
            self.make_numeric_class(self.train_df)
            self.make_numeric_class(self.val_df)
        else:
            self.all_labels = None
            pass

       
    
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

        if self.all_labels is not None:
            all_labels = self.all_labels
        else:
            all_labels = dataframe['class'].unique()

        for index, row in dataframe.iterrows():
            idx = np.where(all_labels== row['class'])[0][0]
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

        if self.all_labels is not None:
            all_labels = self.all_labels
        else:
            all_labels = df['class'].unique()

        # Ensure that both train and val sets have all unique classes
        for unique_class in all_labels:
            if unique_class not in train_df['class'].unique():
                additional_samples = val_df[val_df['class'] == unique_class].sample(frac=0.5, random_state=random_state)
                train_df = train_df.append(additional_samples)
                val_df = val_df.drop(additional_samples.index)
        
        return train_df, val_df
    
    def move_files_to_val_directory(self,valdf, source_directory, destination_directory):
        # Create the output directory if it doesn't exist
        os.makedirs(destination_directory, exist_ok=True)
       
        if os.path.exists(destination_directory) and os.listdir(destination_directory):
            print('validation images already exist, skipping...')
            return 

        # Iterate through the DataFrame and move files to the 'val' directory
        for index, row in valdf.iterrows():
            source_path = os.path.join(source_directory, row['filename'])
            destination_path = os.path.join(destination_directory, row['filename'])
            
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
            else:
                print('Image file does not exist')
            

    
    def Yolo_labels_maker(self,dataframe,output_dir,skip=False,yolo_format=False,keep =False):
        if not skip:
        # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Remove any existing files in the output directory
            if not keep:    
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            # Iterate through the DataFrame and create YOLO label files    
            for index, row in dataframe.iterrows():
                # Replace the extension with ".txt"
                filename = os.path.join(output_dir, os.path.splitext(row['filename'])[0] + ".txt")
                with open(filename, 'a') as file:
                    if not yolo_format:
                        # Preprocess the bounding box (bbox) values by dividing by 1024
                        bbox = [float(value) for value in row['bbox']]
                        y1, x1, y2, x2 = bbox

                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        center_x = x1 + bbox_width / 2
                        center_y = y1 + bbox_height / 2

                        # Normalize the values
                        norm_center_x = center_x / 1024
                        norm_center_y = center_y / 1024
                        norm_bbox_width = bbox_width / 1024
                        norm_bbox_height = bbox_height / 1024
                    
                        file.write(f"{row['num_labels']} {(norm_center_x)} {(norm_center_y)} {(norm_bbox_width)} {(norm_bbox_height)}")
                    elif yolo_format:
                        bbox = [value for value in row['bbox']]
                        xcenter, ycenter, width, height = bbox
                        file.write(f"{row['num_labels']} {(xcenter)} {(ycenter)} {(width)} {(height)}\n")

            print('yolo-format labels  made..')

  #####################################################################################################################################  
   
    def splits_and_yolo_format(self, train_data_pd,test_data_pd,skip=False):
        # Split training data into train and val sets
        new_train_data_pd, val_data_pd = self.make_train_val_dfs(train_data_pd)

        if not skip:
            # Move val images to val directory
            self.move_files_to_val_directory(val_data_pd, "/home/users/maali/Computer_vision_SOC/data/images/train", "/home/users/maali/Computer_vision_SOC/data/images/val")

            # Make labels directories in YOLO format
            self.Yolo_labels_maker(new_train_data_pd, '/home/users/maali/Computer_vision_SOC/data/labels/train')
            self.Yolo_labels_maker(val_data_pd, '/home/users/maali/Computer_vision_SOC/data/labels/val')
            self.Yolo_labels_maker(test_data_pd, '/home/users/maali/Computer_vision_SOC/data/labels/test')
        
        return  new_train_data_pd,val_data_pd
    

    
    def show_image_per_class(self,df,directory_path,sample=1,classname=None,yolo_box=False,save=None,box_thickness=3,yolo_box_ready=False):

        def display_group(group_sample):
                for index, row in group_sample.iterrows():
                        # Extract the image file name and bounding box coordinates
                        file_name = row['filename']
                        y1, x1, y2, x2 = row['bbox']
                        
                        if yolo_box:
                             x1 = y1 - (y2 / 2) 
                             y1 = x1 - (x2 / 2)
                        
                        if  yolo_box_ready:
                            
                            y1*=800
                            
                            y2*=800 
                            
                            x1*=600 
                            
                            x2*=600
                            
                            lower_left_x = y1 - y2/2
                            lower_left_y = x1 - x2/2
                        
                        # Open the image file
                        image_path = os.path.join(directory_path, file_name)
                        with Image.open(image_path) as img:
                                fig, ax = plt.subplots(1)
                                ax.imshow(img)
                                if  yolo_box_ready:
                                    rect = patches.Rectangle((lower_left_x, lower_left_y), y2, x2, linewidth=box_thickness, edgecolor='y', facecolor='none',)
                                # Create a Rectangle patch
                                else:
                                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=box_thickness, edgecolor='g', facecolor='none',)
                                
                                # Add the patch to the Axes
                                ax.add_patch(rect)
                                
                                # Display the image with the bounding box
                                if save is not None:
                                    plt.savefig('/home/users/maali/Computer_vision_SOC/ProcessedImages/plots/train/plot.png')
                                plt.show()
                                plt.close()
        
        df = df[df['class'] != 'no detection']
        
        # Group by the 'class' column
        grouped = df.groupby('class')
        group_sample =''
        for class_name, group in grouped:
            
            if classname is not None and classname == class_name:    
                print(f"Displaying images for class: {class_name}")
                group = group[group['class'] == classname ]
                group_sample = group.sample(n=min(sample, len(group)), random_state=1)
                display_group(group_sample)
                
            
            elif classname is None:
                print(f"Displaying images for class: {class_name}")
                group_sample = group.sample(n=min(sample, len(group)), random_state=1)
                display_group(group_sample)
                