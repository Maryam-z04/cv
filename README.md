# Final Vision Model  

This repository contains the implementation of a deep learning-based object detection model using **InceptionResNetV2** architecture for bounding box prediction. The model is trained to detect and localize objects in images and outputs precise bounding box coordinates for the identified regions.  

# Features  

- Pretrained Backbone**: Utilizes the InceptionResNetV2 architecture pretrained on ImageNet for feature extraction.  
- Custom Head**: A custom regression head for predicting bounding box coordinates.  
- Normalization**: Handles input images by resizing and normalizing pixel values.  
- Bounding Box Scaling**: Outputs denormalized bounding box coordinates for visualization.  
- Visualization**: Includes functionality to draw bounding boxes on the original image.  


# Prerequisites  

To run the notebook, ensure you have the following installed:  

- Python 3.7 or above  
- TensorFlow 2.x  
- OpenCV  
- NumPy  
- Matplotlib  

Install the dependencies using:  

```bash  
pip install tensorflow opencv-python-headless numpy matplotlib  
```  

# Usage  

1. Prepare the Dataset
   - Ensure your dataset consists of images and their respective bounding box annotations.  
   - Update the paths in the notebook to point to your dataset directory.  

2. Training 
   - The model is trained to minimize the difference between predicted bounding box coordinates and ground truth annotations.  
   - Customize hyperparameters (batch size, learning rate, etc.) as needed in the training section.  

3. Prediction and Visualization 
   - Use the `object_detection()` function to predict bounding boxes for new images.  
   - Bounding boxes are visualized on the original image for easy interpretation.  

4. Run the Notebook 
   - Open the `FinalVisionModel.ipynb` in Google Colab.  
   - Follow the sections to preprocess data, train the model, and make predictions.  

# Example  

```python  
path = '/path/to/image.jpg'
image, coords = object_detection(path)
print("Predicted Bounding Box:", coords)
```  

# Results  

The model predicts bounding boxes as normalized coordinates, which are then denormalized to match the original image dimensions.  

# Contributing  

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please submit an issue or a pull request.  

# License  

This project is licensed under the MIT License. See the `LICENSE` file for details.  
