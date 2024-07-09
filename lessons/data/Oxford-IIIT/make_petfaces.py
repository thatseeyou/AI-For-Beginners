#
# This script extracts faces of cats and dogs from the Oxford-IIIT Pet dataset
# and resizes them to 92x92 pixels. The extracted faces are saved in a new directory.
# 
# Download from: https://www.robots.ox.ac.uk/~vgg/data/pets/
#
# Usage:
# 1. Download the images and annotations from the link above
# 2. Extract the images and annotations to the same directory
# 3. Run the script
#

import os
import cv2
import xml.etree.ElementTree as ET

def extract_faces_from_images(image_dir, annotation_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(annotation_dir):
        print(f"Processing {filename}")

        if filename.endswith(".xml"):
            tree = ET.parse(os.path.join(annotation_dir, filename))
            root = tree.getroot()
            
            image_filename = root.find('filename').text
            image_path = os.path.join(image_dir, image_filename)
            image = cv2.imread(image_path)

            for obj in root.iter('object'):
                name = obj.find('name').text
                if name == 'cat' or name == 'dog':
                    breed_name = "_".join(image_filename.split('_')[:-1])
                    breed_dir = os.path.join(output_dir, f"{name}_{breed_name}")
                    if not os.path.exists(breed_dir):
                        os.makedirs(breed_dir)
            
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    face = image[ymin:ymax, xmin:xmax]
                    face_resized = cv2.resize(face, (92, 92))
                    
                    output_filename = os.path.join(breed_dir, f"{os.path.splitext(image_filename)[0]}.jpg")

                    cv2.imwrite(output_filename, face_resized)
                    print(f"Saved face image to {output_filename}")

image_directory = 'images'
annotation_directory = 'annotations/xmls'
output_directory = 'petfaces'

extract_faces_from_images(image_directory, annotation_directory, output_directory)
