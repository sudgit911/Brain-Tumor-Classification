
# Brain Tumor Classification Using VGG16

This project implements a brain tumor classification model using the VGG16 architecture. The model is trained to differentiate between four types of brain tumors: glioma, meningioma, no tumor, and pituitary tumor.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
Brain tumor classification is crucial for medical diagnosis and treatment planning. This project leverages the power of deep learning, specifically the VGG16 architecture, to classify brain tumors from MRI images.

## Dataset
The dataset used in this project is from [Brain-Tumor-Classification-DataSet](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet). It contains MRI images divided into training and testing sets.

Clone the dataset repository:
```bash
git clone https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet
```

## Requirements
Install the necessary packages:
```bash
pip install -r requirements.txt
```

## Project Structure
- `Brain_Tumor_Classification_Using_VGG16_model.ipynb`: Jupyter notebook containing the code for the project.
- `requirements.txt`: List of required packages.
- `README.md`: Project documentation.

## Model Architecture
The model uses the VGG16 architecture with the following layers:
- Convolutional layers
- MaxPooling layers
- Fully connected layers
- Softmax activation for classification

## Training the Model
The model is trained using the following steps:

1. **Import Libraries**: Import necessary libraries for data processing and model building.
2. **Data Preprocessing**: Resize images, normalize pixel values, and split the data into training and testing sets.
3. **Model Compilation**: Compile the model using an optimizer, loss function, and evaluation metrics.
4. **Callbacks**: Use callbacks like EarlyStopping, ReduceLROnPlateau, TensorBoard, and ModelCheckpoint to improve training.
5. **Training**: Train the model on the training data.

```python
import numpy as np
from PIL import Image
import cv2
import os
import random
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
```

## Evaluation
Evaluate the model using classification reports and confusion matrices.

```python
from sklearn.metrics import classification_report, confusion_matrix
```

## Usage
Run the Jupyter notebook to train and evaluate the model. Make sure to update the paths to the dataset directories.

## Results
The results of the model, including accuracy, loss, and evaluation metrics, will be displayed after training.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License
This project is licensed under the MIT License.
