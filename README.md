

# Space Object Detection Notebook description (CO_notebook.ipynb)

This notebook provides a comprehensive solution for training, testing, and validating the yolo model for space object detection, including data preparation, augmentation, model training, predictions, and performance analysis.

## Setup
- **Python Version**: Specify the Python version used.
- **Required Libraries**: `pandas`, `numpy`, `os`, `OpenCV`, `Ultralytics YOLO`.
- **Custom Modules**: `Preprocessor`, `Sampler`, `ObjectDetector`, `ImageProcessor`, `YOLO`.

## Data Preparation
- Define paths for training, validation, and testing images.
- Process CSV files with data annotations.
- Use the `Preprocessor` class for preparing data, splitting datasets, and moving images.

## Sampling
- Utilize the `Sampler` class to create balanced samples from the dataset.

## Data Augmentation
- Employ the `ImageProcessor` class for augmentation tasks.
- Perform background and/or crop-flip augmentation on the training dataset.
- Generate augmented data labels in YOLO format.

## Model Training
- Create an `ObjectDetector` instance for training the model.
- Parameters: epochs, patience, image size, batch size, learning rate, optimizer.

## Testing and Validation
- Evaluate the model on IoU and classification accuracy.
- Make predictions using the trained model on the test dataset.
- Generate a detailed classification report with per-class statistics.
- Calculate and analyze IoU scores for each class.
- Visualize prediction bounding boxes.

## Limitations Analysis
- Analyze and visualize false positives, no detections, and miss-classifications.
- Provide statistical analysis of model limitations across all classes.

## Configuration
- Use `config.yaml` for specifying model and dataset configurations.

## Usage
- Instructions for running each section are included in the notebook.
- Follow the notebook in sequential order for replication.

## Repository Link
- GitHub Repository: [Spark-spacecraft-detection-Yolov8](https://github.com/AliMO77/Spark-spacecraft-detection-Yolov8)

## Additional Notes
- Ensure correct setting of paths as per your local environment.
- The notebook assumes the existence of specified directories and files.
- The trained model is accessible in the /train2(best_model)/weights/best.pt, one can use this to evaluate/predict with this model. 
- The /val2 directory contains the validation stats and /predict3 directory some example inference made with the trained model. 



