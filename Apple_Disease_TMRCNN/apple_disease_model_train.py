import os
from os import listdir
import xml.etree
from xml.etree import ElementTree
from numpy import zeros, asarray

import tmrcnn.utils
import tmrcnn.config
import tmrcnn.model

class AppleDataset(tmrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Defining Class Labels for Apple Diseases
        self.add_class("dataset", 1, "Black Rot")
        self.add_class("dataset", 2, "Apple Scab")
        self.add_class("dataset", 3, "Cedar Apple Rust")

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        # find all images
        for filename in listdir(images_dir):
            print(filename)
			# extract image id
            image_id = filename[:-4]
			#print('IMAGE ID: ',image_id)
			
			# skip all images after 115 if we are building the train set
            if is_train and int(image_id) >= 450:
                continue
			# skip all images before 115 if we are building the test/val set
            if not is_train and int(image_id) < 750:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [0,1,2,3])


    # Extracting bounding boxes from the annotated xml files
    def extract_boxes(self, filename):
		# load and parse the file
        tree = ElementTree.parse(filename)
		# get the root of the document
        root = tree.getroot()
		# extract each bounding box
        boxes = list()
        for box in root.findall('.//object'):
            name = box.find('name').text   #Add label name to the box list
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
		# extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        print("Annotation Path:", len(path))

        boxes, w, h = self.extract_boxes(path)


        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]

            if (box[4]=="Black Rot"):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('Black Rot'))
            elif (box[4]=="Apple Scab"):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('Apple Scab'))
            elif (box[4]=="Cedar Apple Rust"):
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('Cedar Apple Rust'))
        return masks, asarray(class_ids, dtype='int32')



class AppleConfig(tmrcnn.config.Config):
    NAME = "apple_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 1+3

    STEPS_PER_EPOCH = 400

# Train
train_dataset = AppleDataset()
train_dataset.load_dataset(dataset_dir='Apple_dataset', is_train=True)
train_dataset.prepare()
print('Train: %d' % len(train_dataset.image_ids))

# Validation
validation_dataset = AppleDataset()
validation_dataset.load_dataset(dataset_dir='Apple_dataset', is_train=False)
validation_dataset.prepare()
print('Test:%d' % len(validation_dataset.image_ids))

# Model Configuration
Apple_config = AppleConfig()
#Apple_config.display()

# Defining the Mask R-CNN Model Architecture
model = tmrcnn.model.TMaskRCNN(mode='training', 
                             model_dir='./', 
                             config=Apple_config)

model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=Apple_config.LEARNING_RATE, 
            epochs=10, 
            layers='heads')

model_path = 'Apple_disease_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
