# Transformer-MaskRCNN for Apple Tree Disease Instance Segmentation

![Apple Tree Diseases](/path/to/sample_image.jpg)

## Project Description
Enhanced Mask R-CNN model for apple disease instance segmentation by integrating Transformer's multihead-attention mechanism for improved object detection accuracy.

## Sample Images

![Sample 1](/path/to/sample_image_1.jpg)
![Sample 2](/path/to/sample_image_2.jpg)
![Sample 3](/path/to/sample_image_3.jpg)

## User Installation

To install the Transformer-MaskRCNN library, follow these steps:

1. Clone the repository:
git clone https://github.com/muneebelahimalik/Transformer-MaskRCNN

2. Install required packages:
pip install -r requirements.txt


3. Install the library (tmrcnn):
python setup.py install

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

## Credits

Credit goes to [Matter,Inc](https://github.com/matterport) for orignal implementation of [Mask RCNN](https://github.com/matterport/Mask_RCNN)

