import os
import pandas as pd
import shutil
import numpy as np

class Sampler:
    
   def __init__(self,new_train_data_pd,val_data_pd,test_data_pd,train=100,val=20,test=20):

        self.sampled_train_df = self.sample_dataframe_by_class(new_train_data_pd, 'class', train)
        self.sampled_val_df = self.sample_dataframe_by_class(val_data_pd, 'class', val)
        self.sampled_test_df = self.sample_dataframe_by_class(test_data_pd, 'class', test)

        pass

   def sample_dataframe_by_class(self,input_df, class_column, samples_per_class):
        # Group the DataFrame by the specified class column
        np.random.seed(42)
        grouped = input_df.groupby(class_column)
        
        # Initialize an empty DataFrame to store the sample
        sampled_df = pd.DataFrame()

        # Sample rows from each group
        for name, group in grouped:
            if len(group) >= samples_per_class:
                sampled_group = group.sample(samples_per_class)
                sampled_df = pd.concat([sampled_df, sampled_group])

        # Reset the index of the final sampled DataFrame
        sampled_df = sampled_df.reset_index(drop=True)

        return sampled_df

    

   
   def copy_images_and_labels(self,dataframe, source_image_dir, target_image_dir, label_dir, target_label_dir):
    
        # Create the target image directory if it doesn't exist
        os.makedirs(target_image_dir, exist_ok=True)

        # Create the target label directory if it doesn't exist
        os.makedirs(target_label_dir, exist_ok=True)

        # Clear the target directories by removing existing files
        for filename in os.listdir(target_image_dir):
            file_path = os.path.join(target_image_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for filename in os.listdir(target_label_dir):
            file_path = os.path.join(target_label_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            
        for index, row in dataframe.iterrows():
            image_filepath = os.path.join(source_image_dir, row['filename'])
            image_filename = row['filename']

            target_image_path = os.path.join(target_image_dir, image_filename)

            # Copy the image file to the target image directory
            shutil.copy(image_filepath, target_image_path)

            # Check if there is a corresponding label file with the same name (excluding extension)
            label_filename = os.path.splitext(row['filename'])[0] + ".txt"
            label_filepath = os.path.join(label_dir, label_filename)

            if os.path.exists(label_filepath):
                target_label_path = os.path.join(target_label_dir, label_filename)
                # Copy the label file to the target label directory
                shutil.copy(label_filepath, target_label_path)
   
   def make_samples(self, skip=False):
       
       if not skip:
            # Sample the training set
            # sampled_train_df = self.sample_dataframe_by_class(new_train_data_pd, 'class', train)
            
            source_image_directory = '/home/users/maali/Computer_vision_SOC/data/images/train/'
            label_directory = '/home/users/maali/Computer_vision_SOC/data/labels/train/'
            target_image_directory = '/home/users/maali/Computer_vision_SOC/samples/train/images/'
            target_label_directory = '/home/users/maali/Computer_vision_SOC/samples/train/labels/'
            
            self.copy_images_and_labels(self.sampled_train_df, source_image_directory, target_image_directory, label_directory, target_label_directory)
            print('training samples made...')
            # Sample the validation set
            # sampled_val_df = self.sample_dataframe_by_class(val_data_pd, 'class', val)
            
            val_source_image_directory = '/home/users/maali/Computer_vision_SOC/data/images/val/'
            val_label_directory = '/home/users/maali/Computer_vision_SOC/data/labels/val/'
            val_target_image_directory = '/home/users/maali/Computer_vision_SOC/samples/val/images/'
            val_target_label_directory = '/home/users/maali/Computer_vision_SOC/samples/val/labels/'
            self.copy_images_and_labels(self.sampled_val_df, val_source_image_directory, val_target_image_directory, val_label_directory, val_target_label_directory)
            print('validation samples made...')
            # Sample the test set
            # sampled_test_df = self.sample_dataframe_by_class(test_data_pd, 'class', test)
            test_source_image_directory = '/home/users/maali/Computer_vision_SOC/data/images/test/'
            test_label_directory = '/home/users/maali/Computer_vision_SOC/data/labels/test/'
            test_target_image_directory = '/home/users/maali/Computer_vision_SOC/samples/test/images/'
            test_target_label_directory = '/home/users/maali/Computer_vision_SOC/samples/test/labels/'
            self.copy_images_and_labels(self.sampled_test_df, test_source_image_directory, test_target_image_directory, test_label_directory, test_target_label_directory)
            print('testing samples made...')
            
