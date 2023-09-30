# Transformer-MaskRCNN for Apple Tree Disease Instance Segmentation

## Project Description
Enhanced Mask R-CNN model for apple disease instance segmentation by integrating Transformer's multihead-attention mechanism for improved object detection accuracy.

## Model Architecture

![Architecture](https://github.com/muneebelahimalik/Transformer-MaskRCNN/assets/59524535/f31a3952-cc3f-4776-be22-1b172471ce78)

## User Installation

To install the Transformer-MaskRCNN library, follow these steps:

1. Clone the repository:
git clone https://github.com/muneebelahimalik/Transformer-MaskRCNN

2. Install required packages:
pip install -r requirements.txt

3. Install the library (tmrcnn):
python setup.py install

## Folder Structure 
```
Transformer-MaskRCNN/
│
├─ tmrcnn/
│   ├── model.py
│   ├── utils.py
│   ├── config.py
│   ├── visual.py
│   └── __init__.py
│
├─ Apple_Disease_TMRCNN/
│   ├── apple_disease_model_train.py
│   ├── Apple_Disease_inference_model.py
│   ├── reorder_files_in_dataset.py
│   
├──dataset/
│   ├── Apple_dataset.zip
│   └── ...
|
├─ test_images/
│   ├── unseen_1.jpg
│   ├── unseen_2.jpg
│   └── ...
│
├─ requirements.txt
├─ setup.py
└─ README.md

```
## Requirements

- Python 3.6 or higher
- TensorFlow 2.x
- NumPy
- OpenCV
- Matplotlib
- Pillow

## Usage

To use the Transformer-MaskRCNN model for apple tree disease instance segmentation, follow these steps:

1. Prepare your dataset: Organize your apple leaf images and corresponding annotations in the dataset folder.

2. Configure the model: Modify the configuration parameters in the config.py file to suit your dataset and model preferences.

3. Train the model: Run the training script from the Apple_Disease_TMRCNN folder to train the Transformer-MaskRCNN model on your dataset.

4. Evaluate the model: Use the evaluation script to assess the model's performance on a separate validation set.

5. Make predictions: Apply the trained model to new images in the test_images folder using the prediction script.

## COCO Weights

The model was trained on the COCO Weights which you can download from the Google Drive link below
[COCO Weights](https://drive.google.com/file/d/1hHV_eodAH8QEpNEH_gTP47UJ4h93HAB7/view?usp=sharing)

## Trained Models

In case you want to run my trained model, download it from the Google Drive link below.
[TMask-RCNN Trained Model](https://drive.google.com/file/d/1rCWxYvmYygt9ToTQK3TC2D2ov_XRvDuf/view?usp=sharing)


## Credits

Credit goes to [Matter,Inc](https://github.com/matterport) for orignal implementation of [Mask RCNN](https://github.com/matterport/Mask_RCNN)
