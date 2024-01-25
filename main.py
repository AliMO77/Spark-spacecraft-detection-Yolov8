import sys
import os
from Preprocessor import Preprocessor
from Sampler import Sampler 
from ObjectDetection import ObjectDetector 
from imageProcessor import ImageProcessor 
from ultralytics import YOLO

def main():


########################################Data Preparation######################################################

    curr_dir ='/home/users/maali/Computer_vision_SOC'
    source ='/home/users/maali/Computer_vision_SOC/source'
    sys.path.append(curr_dir)
    sys.path.append(source)

    train_image_folder = curr_dir+'/data/images/train/'
    val_image_folder = curr_dir+'/data/images/val/'
    test_image_folder = curr_dir+'/data/images/test/'
    train_data_csv = curr_dir+'/backup/labels/train.csv'
    test_data_csv = curr_dir+'/backup/labels/val.csv'

    processor = Preprocessor(train_image_folder,val_image_folder,train_data_csv,test_data_csv)
    all_labels = processor.all_labels

    train_data_pd,test_data_pd = processor.train_df,processor.val_df
    new_train_data_pd,val_data_pd = processor.make_train_val_dfs(train_data_pd)

    #move val images to val directory 
    processor.move_files_to_val_directory(val_data_pd,source_directory= curr_dir+"/data/images/train",destination_directory =curr_dir+"/data/images/val")

    #make labels directories in yolo format
    processor.Yolo_labels_maker(new_train_data_pd,curr_dir+'/data/labels/train',skip=True)
    processor.Yolo_labels_maker(val_data_pd,curr_dir+'/data/labels/val',skip=True)
    processor.Yolo_labels_maker(test_data_pd,curr_dir+'/data/labels/test',skip=True)


    ########################################Sampling############################################################
    sampler = Sampler(new_train_data_pd, val_data_pd, test_data_pd,train=100,val=50,test=50)
    sampler.make_samples(skip=True)

    ########################################Augmentation########################################################
    img_processor = ImageProcessor(sampler.sampled_train_df,sampler.sampled_val_df,sampler.sampled_test_df,)

    bg_aug_train_df = img_processor.new_backgrounds_augment(sampler.sampled_train_df,skip=True)
    processor.Yolo_labels_maker(bg_aug_train_df,curr_dir+'/samples/train/labels',skip=True,yolo_format=True,keep=True)


    crop_aug_train_df = img_processor.crop_flip_augment(sampler.sampled_train_df,skip = True)
    processor.Yolo_labels_maker(crop_aug_train_df,curr_dir+'/samples/train/labels',skip=True,yolo_format=True,keep=True)

    ########################################Training############################################################

    detector = ObjectDetector(data = curr_dir+'/config.yaml', model ='/home/users/maali/Computer_vision_SOC/runs/detect/train10/weights/best.pt' )

    detector.train_yolo_model(epochs=150,patience =15,imgsz = 256, batch = 8,lr0=0.001 ,optimizer = 'SGD') 
                        
if __name__ == "__main__":
    main()