import os
import glob
import random
import xml.etree.ElementTree as ET

images_dir = 'Apple_dataset/images'
annot_dir = 'Apple_dataset/annots'

imagepath_list = glob.glob(os.path.join(images_dir, '*.jpg'))
random.Random(40).shuffle(imagepath_list)

#padding=len(str(len(imagepath_list)))
padding=1

for n,filepath in enumerate(imagepath_list,1):
    os.rename(filepath, os.path.join(images_dir, '{:>0{}}.jpg'.format(n, padding)))

annotpath_list = glob.glob(os.path.join(annot_dir, '*.xml'))
random.Random(40).shuffle(annotpath_list)
for m,filepath in enumerate(annotpath_list,1):
    os.rename(filepath, os.path.join(annot_dir, '{:>0{}}.xml'.format(m, padding)))

"""
def update_path_in_annotations(imagepath_list, annotpath_list):
    for image_path, annot_path in zip(imagepath_list, annotpath_list):
        image_filename = os.path.basename(image_path)
        image_id, image_ext = os.path.splitext(image_filename)
        xml_filename = image_id + '.xml'

        if not os.path.exists(annot_path):
            continue

        tree = ET.parse(annot_path)
        root = tree.getroot()

        path_tag = root.find('path')
        if path_tag is not None:
            path_tag.text = os.path.abspath(image_path).replace('\\', '/')

        updated_annot_path = os.path.join(annot_dir, xml_filename)
        tree.write(updated_annot_path)

if __name__ == "__main__":
    update_path_in_annotations(imagepath_list, annotpath_list)
"""
