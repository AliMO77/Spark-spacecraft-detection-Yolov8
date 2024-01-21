import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import cv2
from scipy.stats import norm
from PIL import Image, ImageEnhance, ImageOps




class ImageProcessor:

    currdir = '/home/users/maali/Computer_vision_SOC'

    def __init__(self,train_df,val_df,test_df):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
         
        self.train_images = '/home/users/maali/Computer_vision_SOC/samples/train/images'
        self.val_images = '/home/users/maali/Computer_vision_SOC/samples/val/images'
        self.test_images = '/home/users/maali/Computer_vision_SOC/samples/test/images'

        pass


    def plot_average_gaussian(self, class_name, dataset='train', sample_size=None):

        directory = None
        df = None
        save_path = None
        if dataset == 'train':
            directory = self.train_images
            df = self.train_df
            save_path ='/home/users/maali/Computer_vision_SOC/ProcessedImages/plots/train/'
        elif dataset == 'val':
            directory = self.val_images
            df = self.val_df
            save_path ='/home/users/maali/Computer_vision_SOC/ProcessedImages/plots/val/'
        elif dataset == 'test':
            directory = self.test_images
            df = self.test_df
            save_path ='/home/users/maali/Computer_vision_SOC/ProcessedImages/plots/test/'


        # Step 1: Filter the DataFrame by the specified class
        filtered_df = df[df['class'] == class_name]

        if filtered_df.empty:
            print(f"No images found for class: {class_name}")
            return

        # Step 2: Load and process images
        images = []
        if sample_size is None:
            sample_size = len(filtered_df)  # Use all images by default

        selected_indices = random.sample(range(len(filtered_df)), sample_size)

        for index in selected_indices:
            row = filtered_df.iloc[index]
            image_path = os.path.join(directory, row['filename'])
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)

        # Step 3: Calculate pixel values and standard deviation
        pixel_values = np.concatenate(images).ravel()
        mean = np.mean(pixel_values)
        std_dev = np.std(pixel_values)

        # Step 4: Create a range of values for the x-axis (pixel values)
        x = np.linspace(0, 255, 256)

        # Step 5: Calculate the Gaussian distribution
        pdf = norm.pdf(x, mean, std_dev)

        # Step 6: Plot the Gaussian distribution
        plt.figure(figsize=(8, 6))
        plt.plot(x, pdf, color='red', lw=2)
        plt.title(f"Gaussian Distribution for Class: {class_name}")
        plt.xlabel("Pixel Values")
        plt.ylabel("Probability Density")
        plt.grid(True)
       
        plt.savefig(save_path+class_name)
        plt.show()


    def clear_files_in_directory(self, path):
    # Check if the path is a directory
        if not os.path.isdir(path):
            print(f"The path {path} is not a directory.")
            return
        
        # Flag to check if at least one file was found
        at_least_one_file_found = False
        
        # Walk through the directory
        for root, dirs, files in os.walk(path):
            if files:  # If there are files in the directory
                at_least_one_file_found = True
                # Remove each file
                for name in files:
                    file_path = os.path.join(root, name)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Failed to remove file {file_path}. Reason: {e}")
        
        # If no files were found and deleted, skip the 'directory cleared' message
        if at_least_one_file_found:
            print(f"All files in {path} have been cleared.")
        else:
            print("The directory is already empty or contains no files to clear.")


    def preprocess_spacecraft_images(self,image_path):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or path is incorrect")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        clahe_img = clahe.apply(gray)

        # Step 2: Gaussian Blur
        blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0)

        # Step 3: Thresholding
        _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Morphological Operations
        # Use a structural element that is an ellipse, which should help in connecting parts of the spacecraft
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Perform opening to remove small objects like stars (dot shapes)
        opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
        # Perform closing to close small holes within the spacecraft
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Step 5: Contour Detection - find the largest contour assuming it's the spacecraft
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours by area - adjust the min_area as needed
        min_area = 100  # Minimum area to be considered as the spacecraft, adjust based on your images
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Create an empty mask for drawing the large contours
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, large_contours, -1, (255), thickness=cv2.FILLED)

        # Enhance the spacecraft by masking
        enhanced_img = cv2.bitwise_and(image, image, mask=mask)

        return enhanced_img

    def process_and_copy_images(self,selected_classes,dataset ='train',skip=False):

        if not skip:
        
            if dataset == 'train':
                input_dir = self.train_images
                df = self.train_df
                output_dir ='/home/users/maali/Computer_vision_SOC/ProcessedImages/train/images/'
                lab_output_dir ='/home/users/maali/Computer_vision_SOC/ProcessedImages/train/labels/'

                lab_input_dir = '/home/users/maali/Computer_vision_SOC/samples/train/labels/'
            elif dataset == 'val':
                input_dir = self.val_images
                df = self.val_df
                output_dir ='/home/users/maali/Computer_vision_SOC/ProcessedImages/val/images/'
                lab_output_dir ='/home/users/maali/Computer_vision_SOC/ProcessedImages/val/labels/'
                lab_input_dir = '/home/users/maali/Computer_vision_SOC/samples/val/labels/'


            elif dataset == 'test':
                input_dir = self.test_images
                df = self.test_df
                output_dir ='/home/users/maali/Computer_vision_SOC/ProcessedImages/test/images/'
                lab_output_dir ='/home/users/maali/Computer_vision_SOC/ProcessedImages/test/labels/'
                lab_input_dir = '/home/users/maali/Computer_vision_SOC/samples/test/labels/'



            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)


            # Example usage:
            self.clear_files_in_directory(output_dir)

            selected_classes_set = set(selected_classes)  # Convert to a set for efficient membership tests
            for index, row in df.iterrows():
                filename = row['filename']
                class_name = row['class']

                # Check if the class is selected for processing
                if class_name in selected_classes_set:
                    # Process the image by equalizing its histogram
                    image_path = os.path.join(input_dir, filename)
                
                    enhanced_image = self.preprocess_spacecraft_images(image_path)
                    # Save the image in the output directory
                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, enhanced_image)
                else:
                    # Copy the original image to the output directory
                    input_image_path = os.path.join(input_dir, filename)
                    output_image_path = os.path.join(output_dir, filename)
                    shutil.copy(input_image_path, output_image_path)
            #copy labels too
            self.copy_labels(lab_input_dir, df, lab_output_dir)

 

    def copy_labels(self,input_dir, df, output_dir):
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for index, row in df.iterrows():
            filename = row['filename']
            filename_without_extension = os.path.splitext(filename)[0]

            # Source path (original file with extension)
            source_path = os.path.join(input_dir, filename_without_extension+'.txt')

            # Destination path (file without extension)
            destination_path = os.path.join(output_dir, filename_without_extension+'.txt')

            try:
                # Copy the file from the source to the destination
                shutil.copy(source_path, destination_path)
            except FileNotFoundError:
                print(f"File not found: {source_path}")


    def new_backgrounds_augment(self,df,skip =False,image_dir = '/home/users/maali/Computer_vision_SOC/samples/train/images',output_dir = '/home/users/maali/Computer_vision_SOC/samples/train/images'):

        if not skip:
            background_dir = '/home/users/maali/Computer_vision_SOC/syntheticBackgrounds'
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            new_data = []

            for index, row in df.iterrows():
                # Read the original image
                image_path = os.path.join(image_dir, row['filename'])
                image = cv2.imread(image_path)

                # Crop the object
                y1, x1, y2, x2 = map(int, row['bbox'])
                cropped_object = image[y1:y2, x1:x2]

                # Choose a random background
                background_image_name = np.random.choice(os.listdir(background_dir))
                background_image_path = os.path.join(background_dir, background_image_name)
                background = cv2.imread(background_image_path)

                # Get dimensions
                obj_height, obj_width = cropped_object.shape[:2]
                bg_height, bg_width = background.shape[:2]

                # Resize object if it is larger than the background
                if obj_height > bg_height or obj_width > bg_width:
                    scaling_factor = min(bg_width / obj_width, bg_height / obj_height)
                    new_width = int(obj_width * scaling_factor)
                    new_height = int(obj_height * scaling_factor)
                    cropped_object = cv2.resize(cropped_object, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    obj_height, obj_width = new_height, new_width

                # Choose a random location for the object on the background
                x_offset = np.random.randint(0, max(bg_width - obj_width, 1))
                y_offset = np.random.randint(0, max(bg_height - obj_height, 1))

                # Place object on the background
                background[y_offset:y_offset+obj_height, x_offset:x_offset+obj_width] = cropped_object

                # Save the new image
                new_filename = f"background_augmented_{row['filename']}"
                new_image_path = os.path.join(output_dir, new_filename)
                cv2.imwrite(new_image_path, background)

                # Convert bounding box to YOLO format
                x_center = x_offset + (obj_width / 2)
                y_center = y_offset + (obj_height / 2)
                width = obj_width
                height = obj_height
                yolo_bbox = [x_center/bg_width, y_center/bg_height, width/bg_width, height/bg_height]

                # Add to new dataframe
                new_data.append({
                    'filename': new_filename,
                    'class': row['class'],
                    'bbox': ','.join(map(str, yolo_bbox)),
                    'num_labels': row['num_labels']
                })
            
            # Create new dataframe
            new_df = pd.DataFrame(new_data, columns=['filename', 'class', 'bbox', 'num_labels'])
            new_df['bbox'] = new_df['bbox'].apply(eval)
            new_df['bbox'] = new_df['bbox'].apply(list)

            return new_df
        



 
    def crop_flip_augment(self,df, skip=False):
       
        if not skip:
            image_dir = '/home/users/maali/Computer_vision_SOC/samples/train/images'
            output_dir = '/home/users/maali/Computer_vision_SOC/samples/train/images'
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            new_data = []

            for index, row in df.iterrows():
                # Read the original image
                image_path = os.path.join(image_dir, row['filename'])
                image = Image.open(image_path)

                # Crop the object
                y1, x1, y2, x2 = row['bbox']
                
                cropped_object = image.crop((x1, y1, x2, y2))

               # Apply transformations
                if np.random.rand() > 0.8:
                    cropped_object = ImageOps.mirror(cropped_object)
                if np.random.rand() > 0.8:
                    cropped_object = ImageOps.flip(cropped_object)
                if np.random.rand() > 0.8:
                    cropped_object = cropped_object.rotate(np.random.choice([90, 180, 270], p=[0.33, 0.33, 0.34]), expand=True)
                enhancer = ImageEnhance.Contrast(cropped_object)
                cropped_object = enhancer.enhance(np.random.uniform(0.5, 1.5))

                # Create a new white image to place the cropped object
                new_image = Image.new('RGB', (1024,1024), (255, 255, 255))
                obj_width, obj_height = cropped_object.size

                # Calculate the position to place the cropped object on the white image
                x_offset = (1024 - obj_width) // 2
                y_offset = (1024 - obj_height) // 2

                # Paste the cropped object onto the white image
                new_image.paste(cropped_object, (x_offset, y_offset))

                # Save the new image
                new_filename = f"crop_augmented_{row['filename']}"
                new_image_path = os.path.join(output_dir, new_filename)
                new_image.save(new_image_path)

                # Calculate new bounding box in YOLO format
                x_center = (x_offset + obj_width / 2) / 1024
                y_center = (y_offset + obj_height / 2) / 1024
                width = obj_width / 1024
                height = obj_height /1024
                new_bbox = [x_center, y_center, width, height]

                # Add to new dataframe
                new_data.append({
                    'filename': new_filename,
                    'class': row['class'],
                    'bbox': ','.join(map(str, new_bbox)),
                    'num_labels': row['num_labels']
                })
                
            # Create new dataframe
            new_df = pd.DataFrame(new_data, columns=['filename', 'class', 'bbox', 'num_labels'])
            new_df['bbox'] = new_df['bbox'].apply(eval)
            new_df['bbox'] = new_df['bbox'].apply(list)

            return new_df






