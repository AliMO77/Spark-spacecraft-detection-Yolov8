import os
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json
import random
import shutil
from ultralytics.models.yolo.detect import DetectionTrainer
import matplotlib.patches as patches
from PIL import Image
import warnings


class ObjectDetector:
   
    warnings.simplefilter("ignore", category=FutureWarning)

    def __init__(self,data = '/home/users/maali/Computer_vision_SOC/config.yaml',model ='/home/users/maali/Computer_vision_SOC/runs/detect/train10/weights/best.pt' ):
        self.data = data
        self.model = model 
        pass

    def train_yolo_model(self,epochs=25,patience=5,batch=8,lr0=0.0005,imgsz=640,optimizer='auto',cos_lr=True, resume = False,project='/home/users/maali/Computer_vision_SOC/runs/detect',max_det=1,save_dir=None):

        # Specify the save directory for training runs
        
        model = self.model
        args = dict(
            model= model, 
            data=self.data, 
            imgsz=imgsz,
            project=project,
            epochs=epochs,
            optimizer=optimizer,
            patience=patience,
            batch=16,
            lr0=lr0,
            cos_lr = cos_lr,
            resume = resume,
            max_det=max_det,
            save_dir=save_dir
            )
        
        trainer = DetectionTrainer(overrides=args)
        trainer.train()

    
    def test_predict(self,model ='/home/users/maali/Computer_vision_SOC/runs/detect/train/weights/best.pt',num_images=10,
                     output_directory="/home/users/maali/Computer_vision_SOC/samples/test_predict_samples", 
                     project='/home/users/maali/Computer_vision_SOC/runs/detect/predictions',
    ):

        # Set the directory containing your number images
        directory_path = "/home/users/maali/Computer_vision_SOC/samples/test/images"

        # Set the number of images you want to select
        num_images_to_select = num_images

        # Specify the directory where you want to save the selected images
        output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
                # Remove any existing files in the output directory    
        for file in os.listdir(output_directory):
            file_path = os.path.join(output_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # List all image files in the directory

        image_files = [file for file in os.listdir(directory_path) if file.endswith(".jpg")]  # Change the extension as needed

        # Select a random subset of images
        selected_images = random.sample(image_files, num_images_to_select)

        # Create a list to store the file paths of the selected images
        selected_image_paths = []

        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Copy the selected images to the output directory and store their paths
        for image_file in selected_images:
            source_path = os.path.join(directory_path, image_file)
            destination_path = os.path.join(output_directory, image_file)

            # Copy the file to the output directory
            shutil.copy(source_path, destination_path)

            # Append the destination path to the list
            selected_image_paths.append(destination_path)

        model = YOLO(model)
        model.predict(
        source = output_directory,
        project=project,
    
        save=True)
    

 

    def predict_with_yolo(self, test_images_dir,all_labels,model = '/home/users/maali/Computer_vision_SOC/runs/detect/train/weights/best.pt',imgsiz = 256,device=0,max_det=1):
        # Perform predictions with the YOLO model
        def convert_bbox_to_corners(bbox):
            
            x_center, y_center, width, height = bbox

            x1 = x_center - (width / 2)
            y1 = y_center - (height / 2)
            x2 = x_center + (width / 2)
            y2 = y_center + (height / 2)

            return [y1, x1, y2, x2]
        
        
        
        model = YOLO(model)
        res = model.predict(
            source=test_images_dir,
            stream=True,
            project='/home/users/maali/Computer_vision_SOC/runs/detect/predictions',
            max_det = max_det,
            imgsz = imgsiz,
            batch = 16,
        )
        
        imgs = []
        pred_labels = []
        preds_bbox = []

        for r in res:
            name = os.path.basename(r.path)
            boxes = r.boxes.cpu().numpy()
            if len(boxes.cls) > 0:
                max_conf = boxes.conf.max()
                idx = np.where(boxes.conf == max_conf)[0][0]
                label_idx = int(boxes.cls[idx])
                label = all_labels[label_idx]
                bbox = boxes.xywh[idx]
                bbox = convert_bbox_to_corners(bbox)
                pred_labels.append(label)
                preds_bbox.append(bbox)
            else:   
                    pred_labels.append('no detection')
                    preds_bbox.append('no detection')

            imgs.append(name)

        preds_df = pd.DataFrame({'filename': imgs, 'class': pred_labels,'bbox':preds_bbox})

        return preds_df


        
    def classification_report(self, true_df, predicted_df,all_labels):
        merged_df = pd.merge(true_df, predicted_df, on='filename')

        accuracy = accuracy_score(merged_df['class_x'], merged_df['class_y'])
        # precision, recall, f1_score, support = precision_recall_fscore_support(
        #         merged_df['class_x'],
        #           merged_df['class_y'])  
        
        print(" accuracy : {accuracy} ")
        overall_dict = {
            'Overall Accuracy': round(accuracy  * 100,2),
            # 'precision': round(precision[-1]* 100,2),
            # 'recall': round(recall[-1]* 100,2),
            # 'F1-score': round(f1_score[-1]* 100,2)
        }

        class_wise_stats = {}
        
        for label in all_labels:
                selected_classes = [label]
                filtered_df = merged_df[merged_df['class_x'].isin(selected_classes)].drop('class_y',axis=1)
                merged_on_class = pd.merge(filtered_df, merged_df[['filename','class_y']],on='filename' )

                acc = round(accuracy_score( 
                        merged_on_class['class_x'] ,  
                        merged_on_class['class_y'] ,
                        ) 
                        * 100,2,)
               
                # p, r, f1, _ = precision_recall_fscore_support(
                # merged_on_class['class_x'] ,  
                # merged_on_class['class_y'] ,average=None)
               
                class_wise_stats[label] = {
                        'accuracy' : round(acc,2),
                        # 'precision': round(p.max()*100,2),
                        # 'recall': round(r.max()* 100,2),
                        # 'f1_score': round(f1.max()* 100,2)
            }
        
        combined_dict = {**overall_dict, **class_wise_stats}

        # Define the file path where you want to save the JSON data
        file_path = '/home/users/maali/Computer_vision_SOC/source/performanceResults/classification_results.json'

        # Write the combined dictionary to the file in JSON format
        with open(file_path, 'w') as json_file:
                json.dump(combined_dict, json_file, indent=4)

        return overall_dict,class_wise_stats,merged_df
            

    

    def iou_report(self,true_df,predictions_df,all_labels):

        def calculate_iou(boxA, boxB):
            # Determine the coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            
            # Compute the area of intersection
            interArea = max(0, xB - xA) * max(0, yB - yA)
            
            # Compute the area of both bounding boxes
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            
            # Compute the intersection over union
            iou = interArea / float(boxAArea + boxBArea - interArea)
            
            return iou

        
        merged_df = pd.merge(true_df, predictions_df, on='filename')
        iou_scores = []

        adict = {}
        for actual, pred in zip(merged_df['bbox_x'], merged_df['bbox_y']):
            if pred is None or pred == 'no detection' or actual is None or actual == 'no detection':
                #iou_scores.append('no detection')
                
                pred = [0,0,0,0]
                iou = calculate_iou(actual, pred)
                iou_scores.append(iou)
            else: 
                if type(pred)== str:
                    y1,x1,y2,x2 = eval(pred)
                    pred = [y1,x1,y2,x2]
                iou = calculate_iou(actual, np.array(pred))
                iou_scores.append(iou)

        merged_df['ious'] = iou_scores
        
        
        adict['Overall iou score']= {
            'mean':round(np.mean(iou_scores)*100,2),
            'min': round(min(iou_scores)*100,2),
            'max': round(max(iou_scores)*100,2)            
            }
        for label in all_labels:
            df = merged_df[(merged_df['class_x'] == label) & (merged_df['ious'] != 'no detection')]
            adict[label]={
                'mean':round(df['ious'].mean()*100,2),
                'min':round(df['ious'].min()*100,2),
                'max':round(df['ious'].max()*100,2),
            }


        # Define the file path where you want to save the JSON data
        file_path = '/home/users/maali/Computer_vision_SOC/source/performanceResults/iou_report.json'

        # Write the combined dictionary to the file in JSON format
        with open(file_path, 'w') as json_file:
                json.dump(adict, json_file, indent=4)
        
        return iou_scores



    def collect_false_positives(self,true_df, predictions_df,iou_scores,iou_threshold=0.5):

        df = pd.merge(true_df, predictions_df, on='filename')
        df['ious'] = iou_scores

        detections_df = df[df['bbox_y'] != 'no detection']

        # Identify false positives based on IoU threshold and class mismatch
        false_positives = detections_df[(detections_df['ious'] < iou_threshold) | (detections_df['class_x'] != detections_df['class_y'])]

        false_positives_df = false_positives[['filename', 'class_x', 'bbox_x', 'class_y', 'bbox_y', 'ious']]

        return false_positives_df

  

    def show_false_positives(self ,df, directory_path, sample_size=None,classname=None):

        df = df[df != 'no detection'].dropna()

        if classname is not None:
            df = df[df['class_x'] == classname]
        
        # If sample_size is None, display all images
        if sample_size is not None:
            data_sample = df.sample(n=sample_size)
        else:
            data_sample = df
        
        for _, row in data_sample.iterrows():
            img_path = f"{directory_path}/{row['filename']}"
            img = Image.open(img_path)
            
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            
            # Draw bbox_x
            if row['bbox_x'] != 'no detection':
                y1, x1, y2, x2 = row['bbox_x']
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none', label=row['class_x'])
                ax.add_patch(rect)
                plt.text(x1, y1, row['class_x'], color='green', fontsize=8)

            # Draw bbox_y (predicted bounding box)
            if type(row['bbox_y']) == str:
                y1, x1, y2, x2 = eval(row['bbox_y'])
            else: 
                 y1, x1, y2, x2 = row['bbox_y']

            rect_y = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                    linewidth=1, edgecolor='r', facecolor='none', label=f"Pred: {row['class_y']}")
            ax.add_patch(rect_y)
            plt.text(x1, y1, row['class_y'], fontsize=8, color='red')
            # Generate a random integer between 0 and 99999
            random_number = np.random.randint(0, 100000)

            # Save the figure with the random integer in the filename
            plt.savefig(f'/home/users/maali/Computer_vision_SOC/samples/test_predict_samples/fp{random_number}.png')
            plt.show()
            plt.close()
    
    def miss_stats(self, true_df, predictions_df,false_positives_df,all_labels):
        merged_df = pd.merge(true_df, predictions_df,on='filename')
        no_dections = merged_df[merged_df['class_y']=='no detection']

        tot_nrb_images= len(merged_df)
        tot_nbr_nodetects= len(no_dections)
        all_fp = len(false_positives_df)

        stats = []

        for label in all_labels:
            classdf = merged_df[merged_df['class_x']==label]
            classdf_false_positives = false_positives_df[false_positives_df['class_x']==label]
            missclassifictions = classdf[(classdf['class_x'] != classdf['class_y']) & (classdf['class_y'] != 'no detection')]

            tot_images = len(classdf)
            tot_nodetects = len(classdf[classdf['class_y']=='no detection'])
            tot_fp = len(classdf_false_positives)
            tot_mc = len(missclassifictions)

            fp_string = f"{tot_fp} ({tot_fp * 100 / tot_images:.2f}%)"
            nd_str = f"{tot_nodetects} ({tot_nodetects * 100 / tot_images:.2f}%)"
            mc_str = f"{tot_mc} ({tot_mc * 100 / tot_images:.2f}%)"


            class_stats = pd.DataFrame({'nbr_Images':[tot_images],'false_positives':[tot_fp],'no_detections':[tot_nodetects], 'miss_classification':[tot_mc] , 
                                        'False_positives':[fp_string],'No_detections':[nd_str],'Miss_classification':[mc_str]}
                                        ,index=[label])
            
            stats.append(class_stats)



        all_classes_stats = pd.concat(stats)
        all_classes_stats['max_value'] = all_classes_stats[['False_positives', 'No_detections', 'miss_classification']].max(axis=1)
        all_mc = all_classes_stats['miss_classification'].sum()

        all_classes_stats = all_classes_stats.sort_values(by='max_value',ascending=False).drop(['max_value','false_positives','no_detections','miss_classification'],axis=1)


        all_fp_string = f"{all_fp} ({all_fp * 100 / tot_nrb_images:.2f}%)"
        all_nd_str = f"{tot_nbr_nodetects} ({tot_nbr_nodetects * 100 / tot_nrb_images:.2f}%)"
        all_mc_str = f"{all_mc} ({all_mc * 100 / tot_nrb_images:.2f}%)"

        new_row_df = pd.DataFrame([[tot_nrb_images,all_fp_string,all_nd_str,all_mc_str]], columns=['nbr_Images','False_positives', 'No_detections','Miss_classification'],index=['All'])

        all_classes_stats = all_classes_stats.append(new_row_df)
        return all_classes_stats